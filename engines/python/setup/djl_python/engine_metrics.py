#!/usr/bin/env python
#
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
"""
Serves the engine metrics of this Python worker on a Unix domain socket.

An inference engine that runs inside the worker (vLLM) records Prometheus metrics in the default
prometheus_client registry of the worker process, but nothing serves them. DJL Serving starts the
exporter by passing --metrics-sock-name to djl_python_engine. When a scraper asks, DJL Serving
fetches the text over the socket and answers the scraper on its own port. The worker opens no
network listener.

Only the families whose name starts with one of PREFIXES are exported. Metrics that a custom
model.py or another library registers, and the python_* and process_* families, stay private. A
collector that declares no such family is not called at all, so a collector of a custom model.py
that raises or blocks cannot take the engine families down with it.

GET /metrics[?worker=<label>&name[]=<name>...]
  worker   adds the label worker=<label> to every sample (a sample that already has a worker label
           keeps it as exported_worker)
  name[]   keeps only the families that match: the family name, the family name plus _total, or
           the name of any of its samples. Repeatable.
The answer is always the Prometheus text format, whatever the Accept header says.
"""
import copy
import importlib.util
import itertools
import logging
import os
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlsplit

PREFIXES = ("vllm:", )

LOGGER = logging.getLogger(__name__)

_MAX_DISTINCT_WARNINGS = 16
_server = None
_path = None
_warned = set()


class _Families(object):
    """The only part of a registry that prometheus_client.generate_latest uses."""

    def __init__(self, families):
        self._families = families

    def collect(self):
        return iter(self._families)


def _engine_collectors(registry):
    """
    Returns the collectors of registry that declare a family with an engine name, or None when
    registry does not keep the names of its collectors, which means that every collector is read.

    A collector of someone else (a custom model.py) is not called: if it raises or blocks it would
    take every engine family down with it, and its families are not exported anyway.
    """
    by_name = getattr(registry, "_names_to_collectors", None)
    lock = getattr(registry, "_lock", None)
    if by_name is None or lock is None:
        return None
    collectors = []
    seen = set()
    with lock:
        for name, collector in by_name.items():
            if name.startswith(PREFIXES) and id(collector) not in seen:
                seen.add(id(collector))
                collectors.append(collector)
    return collectors


def _selected(family, names):
    if not names:
        return True
    return (family.name in names or family.name + "_total" in names
            or any(sample.name in names for sample in family.samples))


def _with_worker(family, worker):
    labeled = copy.copy(family)
    samples = []
    for sample in family.samples:
        labels = dict(sample.labels)
        if "worker" in labels:
            labels["exported_worker"] = labels.pop("worker")
        labels["worker"] = worker
        samples.append(sample._replace(labels=labels))
    labeled.samples = samples
    return labeled


def render(registry, query=""):
    """
    Returns the exposition text of the engine families in registry.

    :param registry: the prometheus_client registry to read
    :param query: the URL query string, see the module documentation
    :return: the Prometheus text format, UTF-8 encoded
    """
    from prometheus_client import generate_latest
    params = parse_qs(query)
    names = set(params.get("name[]", ()))
    worker = (params.get("worker") or [None])[0]
    collectors = _engine_collectors(registry)
    if collectors is None:
        collected = registry.collect()
    else:
        collected = itertools.chain.from_iterable(collector.collect()
                                                  for collector in collectors)
    families = [
        family for family in collected
        if family.name.startswith(PREFIXES) and _selected(family, names)
    ]
    if worker:
        families = [_with_worker(family, worker) for family in families]
    return generate_latest(_Families(families))


def _unlink(path):
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        LOGGER.debug("engine metrics cannot remove %s: %s", path, e)


def _warn_once(error):
    key = repr(error)
    if key not in _warned and len(_warned) < _MAX_DISTINCT_WARNINGS:
        _warned.add(key)
        LOGGER.warning("engine metrics render failed: %s", error)


class _Handler(BaseHTTPRequestHandler):
    # One response per connection: the server handles one connection at a time.
    protocol_version = "HTTP/1.0"
    # A client that stalls cannot hold the server for longer than this (seconds).
    timeout = 5

    def do_GET(self):  # pylint: disable=invalid-name
        url = urlsplit(self.path)
        if url.path.rstrip("/") != "/metrics":
            self.send_error(404)
            return
        try:
            # The default registry only: with PROMETHEUS_MULTIPROC_DIR set, a collector over that
            # directory would read the files of every worker and the sums would multiply.
            from prometheus_client import CONTENT_TYPE_LATEST, REGISTRY
            body = render(REGISTRY, url.query)
        except Exception as e:  # pylint: disable=broad-except
            _warn_once(e)
            self.send_error(500)
            return
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # pylint: disable=arguments-differ
        # No access log: DJL Serving logs every line the worker writes.
        pass

    def address_string(self):
        # The peer address of a Unix socket is empty.
        return "unix"


if hasattr(socket, "AF_UNIX"):

    class _Server(socketserver.UnixStreamServer):
        # Scrapes that arrive together wait in this queue, because the server answers one at a
        # time, and a connect to a full queue fails at once. The kernel caps the value at
        # net.core.somaxconn.
        request_queue_size = 1024

        def server_bind(self):
            # bind() creates the file with the mode that the umask leaves, and connect() checks that
            # mode when the call starts, so a chmod after bind() would leave a window before
            # listen() in which another user could connect. Create the file as 0600 from the start.
            previous = os.umask(0o177)
            try:
                super().server_bind()
            finally:
                os.umask(previous)
            try:
                os.chmod(self.server_address, 0o600)
            except OSError:
                _unlink(self.server_address)
                raise

        def handle_error(self, request, client_address):
            # A client that closes early, for example after a timeout, is expected: no stack trace on stderr.
            LOGGER.debug("engine metrics request failed", exc_info=True)
else:
    _Server = None


def start(path):
    """
    Starts the exporter on a Unix socket. Never raises: any failure disables the exporter.

    Call it after logging.basicConfig, so that its log lines reach the worker's stdout.

    :param path: the socket file to create, an existing file is replaced
    :return: the running server, or None when the exporter is not available
    """
    global _server, _path  # pylint: disable=global-statement
    try:
        if _server is not None:
            return _server
        if _Server is None:
            LOGGER.info(
                "engine metrics disabled: Unix sockets are unavailable")
            return None
        if importlib.util.find_spec("prometheus_client") is None:
            LOGGER.info(
                "engine metrics disabled: prometheus_client is not installed")
            return None
        if os.path.lexists(path):
            os.unlink(path)
        server = _Server(path, _Handler)
        try:
            threading.Thread(target=server.serve_forever,
                             name="djl-engine-metrics",
                             daemon=True).start()
        except Exception:
            server.server_close()
            _unlink(path)
            raise
        _server, _path = server, path
        LOGGER.info("engine metrics listening on %s", path)
        return server
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.warning("engine metrics disabled: %s", e)
        return None


def stop():
    """Stops the exporter and removes the socket file. Safe to call at any time and twice."""
    global _server, _path  # pylint: disable=global-statement
    server, path, _server, _path = _server, _path, None, None
    try:
        if server is not None:
            server.shutdown()
            server.server_close()
    except Exception as e:  # pylint: disable=broad-except
        LOGGER.debug("engine metrics stop failed: %s", e)
    if path:
        _unlink(path)
