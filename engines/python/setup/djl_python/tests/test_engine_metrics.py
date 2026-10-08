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
import ast
import http.client
import importlib.util
import io
import logging
import os
import queue
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock
from urllib.parse import urlencode

from djl_python import engine_metrics
from djl_python.arg_parser import ArgParser

HAS_PROMETHEUS = importlib.util.find_spec("prometheus_client") is not None
HAS_UNIX_SOCKET = hasattr(socket, "AF_UNIX")
SETUP_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOGGER_NAME = "djl_python.engine_metrics"

if HAS_PROMETHEUS:
    import prometheus_client
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
    from prometheus_client.core import GaugeMetricFamily
    from prometheus_client.parser import text_string_to_metric_families

needs_prometheus = unittest.skipUnless(HAS_PROMETHEUS,
                                       "prometheus_client is not installed")
needs_unix_socket = unittest.skipUnless(HAS_UNIX_SOCKET,
                                        "Unix sockets are not available")

ODD_LABEL_VALUE = 'x,y"z}\\w\nend'


def make_registry():
    """A registry with the shapes vLLM registers, plus families that must not be exported."""
    registry = CollectorRegistry()
    success = Counter("vllm:request_success",
                      "Count of successfully processed requests.",
                      ["model_name", "engine", "finished_reason"],
                      registry=registry)
    success.labels("m", "0", "stop").inc(3)
    success.labels("m", "0", "length").inc()
    ttft = Histogram("vllm:time_to_first_token_seconds",
                     "Histogram of time to first token in seconds.",
                     ["model_name", "engine"],
                     buckets=[0.1, 1.0],
                     registry=registry)
    ttft.labels("m", "0").observe(0.5)
    running = Gauge("vllm:num_requests_running",
                    "Number of requests in model execution batches.",
                    ["model_name", "engine"],
                    registry=registry)
    running.labels("m", "0").set(2)
    lora = Gauge("vllm:lora_requests_info",
                 "Running stats on lora requests.",
                 ["max_lora", "running_lora_adapters"],
                 registry=registry)
    lora.labels("4", ODD_LABEL_VALUE).set(1)
    owner = Gauge("vllm:labeled_by_owner",
                  "A family that already has a worker label.", ["worker"],
                  registry=registry)
    owner.labels("original").set(1)
    Gauge("custom_requests",
          "Registered by a custom handler.",
          registry=registry).set(1)
    Counter("python_other", "Not an engine family.", registry=registry).inc()
    return registry


def parse(body):
    """Parses exposition text, which also proves that the text is valid."""
    return [
        sample
        for family in text_string_to_metric_families(body.decode("utf-8"))
        for sample in family.samples
    ]


def sample_names(body):
    return {sample.name for sample in parse(body)}


class _UnixConnection(http.client.HTTPConnection):

    def __init__(self, path, timeout=10):
        super().__init__("localhost", timeout=timeout)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self._path)


def read_chunk(client):
    chunk = client.recv(65536)
    if not chunk:
        raise AssertionError("the server closed the connection too early")
    return chunk


def fetch(path, target="/metrics", method="GET", headers=None):
    conn = _UnixConnection(path)
    try:
        conn.request(method, target, headers=headers or {})
        response = conn.getresponse()
        return response.status, dict(response.getheaders()), response.read()
    finally:
        conn.close()


class EngineMetricsTestCase(unittest.TestCase):
    """Gives each test a private socket path and a clean module state."""

    def setUp(self):
        engine_metrics.stop()
        engine_metrics._warned.clear()
        self.tmp = tempfile.mkdtemp(prefix="djl")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.addCleanup(engine_metrics.stop)
        self.path = os.path.join(self.tmp, "m.metrics")

    def use_registry(self, registry):
        patcher = mock.patch.object(prometheus_client, "REGISTRY", registry)
        patcher.start()
        self.addCleanup(patcher.stop)
        return registry


@needs_prometheus
class TestRender(EngineMetricsTestCase):

    def setUp(self):
        super().setUp()
        self.registry = make_registry()

    def test_exports_only_engine_families(self):
        body = engine_metrics.render(self.registry)
        names = sample_names(body)
        self.assertIn("vllm:request_success_total", names)
        self.assertIn("vllm:time_to_first_token_seconds_bucket", names)
        self.assertIn("vllm:num_requests_running", names)
        self.assertEqual([n for n in names if not n.startswith("vllm:")], [])
        self.assertNotIn(b"custom_requests", body)
        self.assertNotIn(b"python_other", body)

    def test_text_is_what_prometheus_client_writes(self):
        registry = CollectorRegistry()
        Counter("vllm:requests", "Requests.", ["engine"],
                registry=registry).labels("0").inc()
        Gauge("vllm:running", "Running.", registry=registry).set(2)
        self.assertEqual(engine_metrics.render(registry),
                         prometheus_client.generate_latest(registry))

    def test_worker_label_is_added_to_every_sample(self):
        body = engine_metrics.render(self.registry, "worker=gpu0")
        samples = parse(body)
        self.assertEqual(len(samples),
                         len(parse(engine_metrics.render(self.registry))))
        self.assertEqual({sample.labels["worker"]
                          for sample in samples}, {"gpu0"})
        names = {sample.name for sample in samples}
        for expected in ("vllm:request_success_total",
                         "vllm:request_success_created",
                         "vllm:time_to_first_token_seconds_bucket",
                         "vllm:time_to_first_token_seconds_sum",
                         "vllm:time_to_first_token_seconds_count",
                         "vllm:time_to_first_token_seconds_created",
                         "vllm:num_requests_running"):
            self.assertIn(expected, names)
        # the values do not change
        total = [
            s for s in samples if s.name == "vllm:request_success_total"
            and s.labels["finished_reason"] == "stop"
        ]
        self.assertEqual([s.value for s in total], [3.0])

    def test_label_values_round_trip(self):
        body = engine_metrics.render(self.registry, "worker=gpu0")
        lora = [s for s in parse(body) if s.name == "vllm:lora_requests_info"]
        self.assertEqual(len(lora), 1)
        self.assertEqual(lora[0].labels["running_lora_adapters"],
                         ODD_LABEL_VALUE)
        # the same for a worker label the caller sends
        worker = 'a,"b"}\\c'
        body = engine_metrics.render(self.registry,
                                     urlencode({"worker": worker}))
        self.assertEqual({s.labels["worker"] for s in parse(body)}, {worker})

    def test_existing_worker_label_is_renamed(self):
        body = engine_metrics.render(self.registry, "worker=gpu1")
        owned = [s for s in parse(body) if s.name == "vllm:labeled_by_owner"]
        self.assertEqual(len(owned), 1)
        self.assertEqual(owned[0].labels, {
            "worker": "gpu1",
            "exported_worker": "original"
        })

    def test_blank_worker_adds_no_label(self):
        for query in ("", "worker=", "other=1"):
            body = engine_metrics.render(self.registry, query)
            self.assertFalse([
                s for s in parse(body) if s.name == "vllm:num_requests_running"
                and "worker" in s.labels
            ], query)

    def test_worker_label_does_not_change_the_registry_objects(self):

        class Cached(object):
            """A collector that hands out the same family object on every scrape."""

            def __init__(self):
                self.family = GaugeMetricFamily("vllm:cached",
                                                "Cached.",
                                                labels=["worker"])
                self.family.add_metric(["original"], 1)

            def collect(self):
                yield self.family

        # like the default registry, which asks a collector for its names when it registers
        registry = CollectorRegistry(auto_describe=True)
        registry.register(Cached())
        for worker in ("a", "b", "c"):
            samples = parse(engine_metrics.render(registry,
                                                  f"worker={worker}"))
            self.assertEqual(samples[0].labels, {
                "worker": worker,
                "exported_worker": "original"
            })

    def test_name_filter_matches_family_total_and_sample_names(self):

        def names_for(*names, encoded=True):
            key = "name%5B%5D" if encoded else "name[]"
            query = "&".join(f"{key}={n.replace(':', '%3A')}" for n in names)
            return sample_names(engine_metrics.render(self.registry, query))

        by_family = names_for("vllm:request_success")
        self.assertEqual(
            by_family,
            {"vllm:request_success_total", "vllm:request_success_created"})
        self.assertEqual(names_for("vllm:request_success_total"), by_family)
        self.assertEqual(names_for("vllm:request_success_created"), by_family)
        self.assertEqual(names_for("vllm:request_success", encoded=False),
                         by_family)
        by_sample = names_for("vllm:time_to_first_token_seconds_bucket")
        self.assertEqual(
            by_sample, {
                "vllm:time_to_first_token_seconds_bucket",
                "vllm:time_to_first_token_seconds_sum",
                "vllm:time_to_first_token_seconds_count",
                "vllm:time_to_first_token_seconds_created"
            })
        self.assertEqual(
            names_for("vllm:request_success", "vllm:num_requests_running"),
            by_family | {"vllm:num_requests_running"})

    def test_name_filter_matches_a_counter_that_has_no_samples_yet(self):
        # a labelled counter has no sample until its first child exists
        Counter("vllm:no_children",
                "No child yet.", ["engine"],
                registry=self.registry)
        for name in ("vllm:no_children", "vllm:no_children_total"):
            body = engine_metrics.render(self.registry, f"name[]={name}")
            self.assertEqual(
                body, b"# HELP vllm:no_children_total No child yet.\n"
                b"# TYPE vllm:no_children_total counter\n")

    def test_name_filter_edge_cases(self):
        everything = engine_metrics.render(self.registry)
        cases = {
            "unknown name": ("name[]=vllm:unknown", b""),
            "name of a custom family": ("name[]=custom_requests", b""),
            "sample name of a custom family":
            ("name[]=python_other_total", b""),
            "blank name": ("name[]=", everything),
        }
        for name, (query, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(engine_metrics.render(self.registry, query),
                                 expected)

    def test_name_filter_and_worker_label_combine(self):
        samples = parse(
            engine_metrics.render(
                self.registry, "worker=gpu2&name[]=vllm:num_requests_running"))
        self.assertEqual([(s.name, s.labels["worker"]) for s in samples],
                         [("vllm:num_requests_running", "gpu2")])

    def test_collectors_without_an_engine_family_are_not_called(self):

        class Foreign(object):
            calls = 0

            def describe(self):
                return []  # no names: register() does not call collect()

            def collect(self):
                Foreign.calls += 1
                raise RuntimeError("a collector of the customer fails")

        self.registry.register(Foreign())
        body = engine_metrics.render(self.registry)
        self.assertIn("vllm:request_success_total", sample_names(body))
        self.assertEqual(Foreign.calls, 0)

        # a function that a gauge calls when it is read is a collector too
        Gauge("custom_function", "Raises when read.",
              registry=self.registry).set_function(lambda: 1 / 0)
        self.assertEqual(engine_metrics.render(self.registry), body)

    def test_collector_that_declares_an_engine_family_is_called_once(self):

        class Engine(object):
            calls = 0

            def describe(self):
                return [
                    GaugeMetricFamily("vllm:declared_a", "A."),
                    GaugeMetricFamily("vllm:declared_b", "B.")
                ]

            def collect(self):
                Engine.calls += 1
                for name in ("vllm:declared_a", "vllm:declared_b"):
                    family = GaugeMetricFamily(name, "Declared.")
                    family.add_metric([], 1)
                    yield family

        self.registry.register(Engine())
        names = sample_names(engine_metrics.render(self.registry))
        self.assertTrue({"vllm:declared_a", "vllm:declared_b"} <= names)
        self.assertEqual(Engine.calls, 1)

    def test_registry_that_keeps_no_collector_names_is_read_whole(self):

        class Plain(object):

            def collect(self):
                # the families that are not engine families are dropped by their name
                for name in ("vllm:plain", "python_plain", "custom_plain"):
                    family = GaugeMetricFamily(name, "Plain.")
                    family.add_metric([], 1)
                    yield family

        self.assertEqual(sample_names(engine_metrics.render(Plain())),
                         {"vllm:plain"})

    def test_mixed_collector_exports_only_its_engine_family(self):
        # collectors are selected by the names they declare, so one that declares an engine family
        # and also yields another family reaches the filter by family name
        class Mixed(object):

            def collect(self):
                for name in ("vllm:mixed", "mixed_private"):
                    family = GaugeMetricFamily(name, "Mixed.")
                    family.add_metric([], 1)
                    yield family

        registry = CollectorRegistry(auto_describe=True)
        registry.register(Mixed())
        body = engine_metrics.render(registry)
        self.assertEqual(sample_names(body), {"vllm:mixed"})
        self.assertNotIn(b"mixed_private", body)


@needs_prometheus
@needs_unix_socket
class TestServer(EngineMetricsTestCase):

    def setUp(self):
        super().setUp()
        self.registry = self.use_registry(make_registry())

    def start(self):
        server = engine_metrics.start(self.path)
        self.assertIsNotNone(server)
        return server

    def test_serves_metrics_over_the_socket(self):
        self.start()
        status, headers, body = fetch(self.path)
        self.assertEqual(status, 200)
        self.assertEqual(int(headers["Content-Length"]), len(body))
        self.assertEqual(headers["Content-Type"],
                         prometheus_client.CONTENT_TYPE_LATEST)
        self.assertEqual(body, engine_metrics.render(self.registry))
        self.assertIn("vllm:request_success_total", sample_names(body))

    def test_socket_file_is_private(self):
        self.start()
        self.assertTrue(stat.S_ISSOCK(os.stat(self.path).st_mode))
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)

    def test_socket_file_is_private_from_the_moment_it_exists(self):
        # connect() checks the mode of the file when the call starts, so a chmod after bind() would
        # leave a window before listen() in which another user could connect. With a permissive
        # umask the file has to be 0600 already when the chmod runs.
        modes = []
        real_chmod = os.chmod

        def spy(path, mode, *args, **kwargs):
            modes.append(stat.S_IMODE(os.stat(path).st_mode))
            return real_chmod(path, mode, *args, **kwargs)

        previous = os.umask(0)
        self.addCleanup(os.umask, previous)
        with mock.patch.object(os, "chmod", spy):
            self.start()
        self.assertEqual(modes, [0o600])
        self.assertEqual(stat.S_IMODE(os.stat(self.path).st_mode), 0o600)
        # the umask of the process is the one it was before
        self.assertEqual(os.umask(0), 0)

    def test_text_format_whatever_the_accept_header_says(self):
        self.start()
        _, plain, plain_body = fetch(self.path)
        for accept in (
                "application/openmetrics-text; version=1.0.0",
                "application/openmetrics-text;version=1.0.0;"
                "escaping=allow-utf-8;q=0.6,text/plain;version=0.0.4;q=0.3",
                "*/*"):
            status, headers, body = fetch(self.path,
                                          headers={"Accept": accept})
            self.assertEqual(status, 200, accept)
            self.assertEqual(headers["Content-Type"], plain["Content-Type"])
            self.assertEqual(body, plain_body)
            self.assertIn(b"# TYPE vllm:request_success_total counter", body)
            self.assertNotIn(b"# EOF", body)
            self.assertNotIn(b"vllm_request_success", body)

    def test_trailing_slash_and_query(self):
        self.start()
        _, _, plain = fetch(self.path)
        status, _, body = fetch(self.path, "/metrics/")
        self.assertEqual((status, body), (200, plain))
        status, _, body = fetch(
            self.path, "/metrics?worker=gpu3&name%5B%5D=vllm%3Anum_requests_"
            "running&ignored=1")
        self.assertEqual(status, 200)
        self.assertEqual([(s.name, s.labels["worker"]) for s in parse(body)],
                         [("vllm:num_requests_running", "gpu3")])

    def test_other_paths_and_methods_are_refused(self):
        self.start()
        for target in ("/other", "/", "/metrics/other", "/server/metrics"):
            self.assertEqual(fetch(self.path, target)[0], 404, target)
        self.assertGreaterEqual(fetch(self.path, method="POST")[0], 400)
        self.assertGreaterEqual(fetch(self.path, method="DELETE")[0], 400)
        # still serving
        self.assertEqual(fetch(self.path)[0], 200)

    def test_registry_changes_show_up_on_the_next_scrape(self):
        self.start()
        gauge = Gauge("vllm:late", "Registered later.", registry=self.registry)
        gauge.set(5)
        samples = parse(fetch(self.path)[2])
        self.assertIn(("vllm:late", 5.0), [(s.name, s.value) for s in samples])
        gauge.set(6)
        samples = parse(fetch(self.path)[2])
        self.assertIn(("vllm:late", 6.0), [(s.name, s.value) for s in samples])

    def test_concurrent_scrapes(self):
        self.start()
        expected = engine_metrics.render(self.registry)
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: fetch(self.path), range(40)))
        self.assertEqual({r[0] for r in results}, {200})
        self.assertEqual({r[2] for r in results}, {expected})

    def test_connection_is_closed_after_each_response(self):
        # The server handles one connection at a time, so a client that asks for keep-alive must
        # not be able to hold it.
        self.start()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.addCleanup(client.close)
        client.settimeout(10)
        client.connect(self.path)
        client.sendall(b"GET /metrics HTTP/1.1\r\nHost: localhost\r\n"
                       b"Connection: keep-alive\r\n\r\n")
        data = b""
        while b"\r\n\r\n" not in data:
            data += read_chunk(client)
        head, _, body = data.partition(b"\r\n\r\n")
        length = int([
            line.split(b":")[1] for line in head.split(b"\r\n")
            if line.lower().startswith(b"content-length")
        ][0])
        while len(body) < length:
            body += read_chunk(client)
        self.assertIn(b" 200 ", head.split(b"\r\n")[0])
        client.settimeout(2)  # the handler's own timeout is 5 seconds
        self.assertEqual(client.recv(1), b"")  # closed by the server
        self.assertEqual(fetch(self.path)[0], 200)

    def test_a_blocking_foreign_collector_cannot_block_the_exporter(self):
        release = threading.Event()
        self.addCleanup(release.set)

        class Blocking(object):

            def describe(self):
                return []

            def collect(self):
                release.wait()
                return iter(())

        self.registry.register(Blocking())
        self.start()
        for _ in range(3):
            self.assertEqual(fetch(self.path)[0], 200)

    def test_limits(self):
        self.assertEqual(engine_metrics._Handler.timeout, 5)
        self.assertEqual(engine_metrics._Server.request_queue_size, 1024)

    def test_scrapes_that_arrive_together_wait_in_the_queue(self):
        # The server answers one connection at a time. While it is busy, a connect has to find room
        # in the queue: a non-blocking connect to a full queue fails at once, which a scraper sees
        # as a failed scrape.
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def busy():
            entered.set()
            release.wait()
            return 1

        Gauge("vllm:busy", "Blocks while it is read.",
              registry=self.registry).set_function(busy)
        self.start()
        clients = []

        def close_all():
            for client in clients:
                client.close()

        self.addCleanup(close_all)
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        clients.append(first)
        first.settimeout(30)
        first.connect(self.path)
        first.sendall(b"GET /metrics HTTP/1.0\r\n\r\n")
        self.assertTrue(entered.wait(30),
                        "the server did not start the scrape")
        for _ in range(40):  # the old queue of 16 had room for 17
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            clients.append(client)
            client.settimeout(30)
            client.connect(
                self.path)  # raises BlockingIOError when the queue is full
            client.sendall(b"GET /metrics HTTP/1.0\r\n\r\n")
        release.set()
        for client in clients:
            data = b""
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                data += chunk
            self.assertTrue(data.startswith(b"HTTP/1.0 200 "), data[:80])

    def test_a_stalled_client_cannot_block_the_server(self):
        with mock.patch.object(engine_metrics._Handler, "timeout", 0.3):
            self.start()
            stalled = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.addCleanup(stalled.close)
            stalled.connect(self.path)  # connects and never sends a request
            started = time.monotonic()
            status, _, _ = fetch(self.path)
            elapsed = time.monotonic() - started
        self.assertEqual(status, 200)
        self.assertLess(elapsed, 5)

    def test_client_that_disconnects_mid_response_writes_nothing_to_stderr(
            self):
        big = Gauge("vllm:big",
                    "A response bigger than the socket buffer.", ["i"],
                    registry=self.registry)
        for i in range(30000):
            big.labels(str(i)).set(i)
        self.start()
        stderr = io.StringIO()
        with mock.patch.object(sys, "stderr", stderr):
            with self.assertLogs(LOGGER_NAME, level="DEBUG") as logs:
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.settimeout(10)
                client.connect(self.path)
                client.sendall(b"GET /metrics HTTP/1.0\r\n\r\n")
                self.assertTrue(client.recv(100))
                client.close()
                # The server handles one connection at a time, so once this one is
                # answered the broken one has been handled.
                status, _, body = fetch(self.path)
        self.assertEqual(status, 200)
        self.assertGreater(len(body), 400000)
        self.assertEqual(stderr.getvalue(), "")
        self.assertTrue([r for r in logs.records if r.exc_info],
                        "the failed request was not logged at debug level")

    def test_render_error_gives_500_and_one_warning(self):

        class Failing(object):

            def __init__(self, message):
                self.message = message

            def collect(self):
                raise RuntimeError(self.message)

        self.use_registry(Failing("first failure"))
        self.start()
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            for _ in range(3):
                self.assertEqual(fetch(self.path)[0], 500)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("first failure", logs.output[0])
        # a different failure is a different message
        self.use_registry(Failing("second failure"))
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            for _ in range(2):
                self.assertEqual(fetch(self.path)[0], 500)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("second failure", logs.output[0])
        # and the exporter recovers
        self.use_registry(self.registry)
        self.assertEqual(fetch(self.path)[0], 200)

    def test_warnings_for_distinct_errors_are_bounded(self):
        limit = engine_metrics._MAX_DISTINCT_WARNINGS
        with self.assertLogs(LOGGER_NAME, level="WARNING") as logs:
            for i in range(limit + 5):
                engine_metrics._warn_once(RuntimeError(f"error {i}"))
        self.assertEqual(len(logs.records), limit)

    def test_multiprocess_dir_is_ignored(self):
        # In this process the output comes from the registry that is patched in as REGISTRY.
        mp_dir = os.path.join(self.tmp, "multiproc")
        os.mkdir(mp_dir)
        with mock.patch.dict(os.environ, {"PROMETHEUS_MULTIPROC_DIR": mp_dir}):
            self.start()
            status, _, body = fetch(self.path)
        self.assertEqual(status, 200)
        self.assertEqual(body, engine_metrics.render(self.registry))
        self.assertEqual(os.listdir(mp_dir), [])


@needs_prometheus
@needs_unix_socket
class TestLifecycle(EngineMetricsTestCase):

    def exporter_threads(self):
        return [
            t for t in threading.enumerate() if t.name == "djl-engine-metrics"
        ]

    def assert_start_is_disabled(self, path, level=logging.WARNING):
        """start() returns None, logs one record and leaves no socket file."""
        with self.assertLogs(LOGGER_NAME, level=level) as logs:
            self.assertIsNone(engine_metrics.start(path))
        self.assertEqual([r.levelno for r in logs.records], [level])
        self.assertIsNone(engine_metrics._server)
        sockets = [
            name for name in os.listdir(self.tmp)
            if stat.S_ISSOCK(os.lstat(os.path.join(self.tmp, name)).st_mode)
        ]
        self.assertEqual(sockets, [])

    def test_start_twice_returns_the_same_server(self):
        server = engine_metrics.start(self.path)
        self.assertIsNotNone(server)
        self.assertIs(engine_metrics.start(self.path), server)
        other = os.path.join(self.tmp, "other")
        self.assertIs(engine_metrics.start(other), server)
        self.assertFalse(os.path.exists(other))
        threads = self.exporter_threads()
        self.assertEqual(len(threads), 1)
        self.assertTrue(threads[0].daemon)

    def test_stale_file_is_replaced(self):

        def stale_socket(path):
            stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stale.bind(path)
            stale.close()  # leaves the file behind, as a killed process does

        def regular_file(path):
            with open(path, "w") as f:
                f.write("stale")

        def dangling_link(path):
            os.symlink(os.path.join(self.tmp, "missing"), path)

        for kind, create in (("socket", stale_socket), ("file", regular_file),
                             ("dangling link", dangling_link)):
            with self.subTest(kind):
                create(self.path)
                self.assertTrue(os.path.lexists(self.path))
                self.assertIsNotNone(engine_metrics.start(self.path))
                self.assertEqual(fetch(self.path)[0], 200)
                engine_metrics.stop()

    def test_start_that_cannot_bind_returns_none_with_one_warning(self):
        os.mkdir(os.path.join(self.tmp, "directory"))
        cases = {
            "path too long for a socket": os.path.join(self.tmp, "x" * 120),
            "missing directory": os.path.join(self.tmp, "missing",
                                              "m.metrics"),
            "directory in the way": os.path.join(self.tmp, "directory"),
        }
        for name, path in cases.items():
            with self.subTest(name):
                self.assert_start_is_disabled(path)
        self.assertTrue(os.path.isdir(cases["directory in the way"]))

    def test_umask_is_restored_when_bind_fails(self):
        previous = os.umask(0o022)
        self.addCleanup(os.umask, previous)
        self.assert_start_is_disabled(os.path.join(self.tmp, "x" * 120))
        self.assertEqual(os.umask(0o022), 0o022)

    @unittest.skipIf(not hasattr(os, "geteuid") or os.geteuid() == 0,
                     "permissions do not apply to root")
    def test_unwritable_directory_returns_none_with_one_warning(self):
        readonly = os.path.join(self.tmp, "readonly")
        os.mkdir(readonly, 0o500)
        self.assert_start_is_disabled(os.path.join(readonly, "m.metrics"))

    def test_failure_to_set_the_mode_leaves_nothing_behind(self):
        # a file that is left behind would be taken for the socket of a working exporter
        with mock.patch.object(os, "chmod", side_effect=PermissionError("no")):
            self.assert_start_is_disabled(self.path)
        self.assertFalse(os.path.lexists(self.path))

    def test_failure_to_start_the_thread_leaves_nothing_behind(self):
        with mock.patch.object(threading.Thread,
                               "start",
                               side_effect=RuntimeError("no threads")):
            self.assert_start_is_disabled(self.path)
        self.assertFalse(os.path.lexists(self.path))

    def test_missing_prometheus_client_returns_none_and_creates_no_file(self):
        with mock.patch("importlib.util.find_spec", return_value=None):
            self.assert_start_is_disabled(self.path, level=logging.INFO)
        self.assertFalse(os.path.lexists(self.path))

    def test_unavailable_unix_sockets_return_none(self):
        # the module sets _Server to None when socket has no AF_UNIX
        with mock.patch.object(engine_metrics, "_Server", None):
            self.assert_start_is_disabled(self.path, level=logging.INFO)

    def test_stop_removes_the_socket_and_is_idempotent(self):
        engine_metrics.stop()  # nothing to stop yet
        engine_metrics.start(self.path)
        self.assertTrue(os.path.exists(self.path))
        engine_metrics.stop()
        self.assertFalse(os.path.lexists(self.path))
        engine_metrics.stop()
        self.assertEqual(self.exporter_threads(), [])
        with self.assertRaises(OSError):
            fetch(self.path)

    def test_stop_tolerates_a_socket_file_that_is_already_gone(self):
        engine_metrics.start(self.path)
        os.unlink(self.path)
        engine_metrics.stop()

    def test_stop_never_raises(self):
        server = engine_metrics.start(self.path)
        with mock.patch.object(server, "shutdown", side_effect=OSError("x")):
            engine_metrics.stop()
        server.shutdown()
        server.server_close()
        self.assertFalse(os.path.lexists(self.path))

    def test_restart_after_stop(self):
        first = engine_metrics.start(self.path)
        engine_metrics.stop()
        second = engine_metrics.start(self.path)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertEqual(fetch(self.path)[0], 200)

    def test_no_python_family_leaks_from_the_default_registry(self):
        # the default registry of this process: standard collectors only, nothing named vllm:
        engine_metrics.start(self.path)
        status, _, body = fetch(self.path)
        self.assertEqual(status, 200)
        self.assertNotIn(b"python_gc", body)
        self.assertNotIn(b"process_", body)
        self.assertNotIn(b"python_info", body)


@needs_prometheus
class TestMultiprocessMode(unittest.TestCase):

    SCRIPT = textwrap.dedent("""
        import socket, sys
        from prometheus_client import Counter
        counter = Counter("vllm:mp_requests", "Requests.")
        counter.inc(float(sys.argv[1]))
        if len(sys.argv) > 2:
            from djl_python import engine_metrics
            assert engine_metrics.start(sys.argv[2]) is not None
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.connect(sys.argv[2])
            client.sendall(b"GET /metrics HTTP/1.0\\r\\n\\r\\n")
            data = b""
            while True:
                chunk = client.recv(65536)
                if not chunk:
                    break
                data += chunk
            engine_metrics.stop()
            print(data.decode())
        """)

    def run_script(self, mp_dir, *args):
        env = dict(os.environ)
        env["PROMETHEUS_MULTIPROC_DIR"] = mp_dir
        env["PYTHONPATH"] = os.pathsep.join([SETUP_DIR] + (
            [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return subprocess.run([sys.executable, "-c", self.SCRIPT, *args],
                              env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              universal_newlines=True,
                              timeout=120)

    @needs_unix_socket
    def test_exports_this_process_only_when_multiprocess_mode_is_on(self):
        # The process imports prometheus_client with the variable set, as a worker would. Another
        # process leaves 5.0 in the directory; a collector over the directory would report 7.0.
        tmp = tempfile.mkdtemp(prefix="djl")
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        mp_dir = os.path.join(tmp, "multiproc")
        os.mkdir(mp_dir)
        other = self.run_script(mp_dir, "5")
        self.assertEqual(other.returncode, 0, other.stdout)
        self.assertTrue([f for f in os.listdir(mp_dir) if f.endswith(".db")],
                        "multiprocess mode was not active")
        result = self.run_script(mp_dir, "2", os.path.join(tmp, "m.metrics"))
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("200 OK", result.stdout)
        self.assertIn("vllm:mp_requests_total 2.0", result.stdout)
        self.assertNotIn("7.0", result.stdout)


class TestImport(unittest.TestCase):

    def run_python(self, code):
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join([SETUP_DIR] + (
            [env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        return subprocess.run([sys.executable, "-c", code],
                              env=env,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,
                              universal_newlines=True,
                              timeout=120)

    def test_import_and_start_do_not_configure_logging(self):
        # A log record sent through the logging module functions before logging.basicConfig
        # installs a stderr handler, and basicConfig then does nothing. The worker's
        # "Python engine started." line must reach stdout.
        result = self.run_python(
            textwrap.dedent("""
                import logging
                from djl_python import engine_metrics
                engine_metrics.start("/nonexistent/directory/m.metrics")
                engine_metrics.stop()
                assert logging.getLogger().handlers == [], logging.getLogger().handlers
                """))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_import_without_unix_sockets(self):
        # Windows has no socket.AF_UNIX and so no socketserver.UnixStreamServer.
        result = self.run_python(
            textwrap.dedent("""
                import socket
                del socket.AF_UNIX
                from djl_python import engine_metrics
                assert engine_metrics._Server is None
                assert engine_metrics.start("/tmp/m.metrics") is None
                engine_metrics.stop()
                """))
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_prometheus_client_is_imported_lazily(self):
        # Workers that do not export metrics, and images without prometheus_client, import the
        # module too.
        with open(engine_metrics.__file__) as f:
            tree = ast.parse(f.read())
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(
                    imported_name.name.split(".")[0]
                    for imported_name in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module.split(".")[0])
        self.assertNotIn("prometheus_client", imported)
        self.assertNotIn("djl_python", imported)


class TestArguments(unittest.TestCase):
    # The arguments Java passes today, element for element.
    ARGS = [
        "--sock-type", "unix", "--sock-name", "/tmp/djl_sock.19000",
        "--model-dir", "/opt/ml/model", "--entry-point", "", "--device-id",
        "0", "--cluster-size", "1", "--recommended-entry-point",
        "djl_python.lmi_vllm.vllm_async_service", "--log-level", "info",
        "--async-mode"
    ]

    def test_metrics_socket_is_optional(self):
        args = ArgParser.python_engine_args().parse_args(self.ARGS)
        self.assertIsNone(args.metrics_sock_name)
        self.assertEqual(args.sock_name, "/tmp/djl_sock.19000")

    def test_metrics_socket_argument(self):
        args = ArgParser.python_engine_args().parse_args(
            self.ARGS + ["--metrics-sock-name", "/tmp/djl_sock.19000.metrics"])
        self.assertEqual(args.metrics_sock_name, "/tmp/djl_sock.19000.metrics")
        self.assertEqual(args.sock_name, "/tmp/djl_sock.19000")
        self.assertTrue(args.async_mode)


MODEL = textwrap.dedent("""
    from prometheus_client import Gauge

    try:
        Gauge("vllm:djl_test_gauge", "A gauge from the worker.").set(7)
        Gauge("custom_djl_test_gauge", "A gauge from a custom handler.").set(8)
    except ValueError:
        pass  # the engine loads model.py a second time to look for formatters


    def handle(inputs):
        return None
    """)

MODEL_WITH_FOREIGN_COLLECTORS = textwrap.dedent("""
    import time
    from prometheus_client import Gauge, REGISTRY


    class Failing(object):

        def describe(self):
            return []

        def collect(self):
            raise RuntimeError("a collector of the customer fails")


    def block():
        time.sleep(3600)


    try:
        Gauge("vllm:djl_test_gauge", "A gauge from the worker.").set(7)
        Gauge("custom_blocking", "A function that never returns.").set_function(block)
        Gauge("custom_failing", "A function that raises.").set_function(lambda: 1 / 0)
        REGISTRY.register(Failing())
    except ValueError:
        pass  # the engine loads model.py a second time to look for formatters


    def handle(inputs):
        return None
    """)


@needs_prometheus
@needs_unix_socket
class TestWorkerProcess(unittest.TestCase):
    """Runs djl_python_engine.py as DJL Serving does, without the Java side."""

    @classmethod
    def setUpClass(cls):
        probe = subprocess.run(
            [sys.executable, "-c", "import djl_python_engine"],
            cwd=SETUP_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            timeout=300)
        if probe.returncode != 0:
            raise unittest.SkipTest("djl_python_engine cannot be imported: " +
                                    probe.stdout[-300:])

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="djl")
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.model_dir = os.path.join(self.tmp, "model")
        os.mkdir(self.model_dir)
        with open(os.path.join(self.model_dir, "model.py"), "w") as f:
            f.write(MODEL)
        self.sock = os.path.join(self.tmp, "djl_sock.19000")
        self.metrics_sock = self.sock + ".metrics"
        self.process = None

    def launch(self, *extra, env=None):
        cmd = [
            sys.executable, "-u",
            os.path.join(SETUP_DIR, "djl_python_engine.py"), "--sock-type",
            "unix", "--sock-name", self.sock, "--model-dir", self.model_dir,
            "--entry-point", "model.py", "--no-async-mode", *extra
        ]
        environment = dict(os.environ)
        environment.update(env or {})
        self.process = subprocess.Popen(cmd,
                                        cwd=self.model_dir,
                                        env=environment,
                                        stdout=subprocess.PIPE,
                                        stderr=subprocess.STDOUT,
                                        universal_newlines=True)
        self.addCleanup(self.kill)
        self.lines = queue.Queue()
        self.log = []

        def pump():
            for line in self.process.stdout:
                self.log.append(line)
                self.lines.put(line)
            self.lines.put(None)

        threading.Thread(target=pump, daemon=True).start()

    def kill(self):
        if self.process is not None and self.process.poll() is None:
            self.process.kill()
        if self.process is not None:
            self.process.wait(timeout=30)
            self.process.stdout.close()

    def wait_for_line(self, text, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                self.fail("the worker exited:\n" + "".join(self.log))
            if text in line:
                return line
        self.fail(f"no line with {text!r} in {timeout} s:\n" +
                  "".join(self.log))

    def test_worker_serves_its_engine_metrics_and_cleans_up(self):
        self.launch("--metrics-sock-name", self.metrics_sock)
        self.wait_for_line("Python engine started.")
        self.assertTrue(stat.S_ISSOCK(os.stat(self.metrics_sock).st_mode))
        self.assertEqual(stat.S_IMODE(os.stat(self.metrics_sock).st_mode),
                         0o600)
        status, _, body = fetch(self.metrics_sock)
        self.assertEqual(status, 200)
        names = sample_names(body)
        self.assertEqual(names, {"vllm:djl_test_gauge"}, body)
        self.assertIn(b"vllm:djl_test_gauge 7.0", body)
        # DJL Serving closes the request socket when it stops the worker
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect(self.sock)
        client.close()
        self.assertEqual(self.process.wait(timeout=60), 1)
        output = "".join(self.log)
        self.assertIn("Python process finished", output)
        self.assertFalse(os.path.lexists(self.metrics_sock), output)
        self.assertFalse(os.path.lexists(self.sock), output)

    def test_worker_ignores_the_collectors_of_the_model(self):
        # A model.py can register collectors that fail or never return. They are not engine
        # families, so the exporter does not call them.
        with open(os.path.join(self.model_dir, "model.py"), "w") as f:
            f.write(MODEL_WITH_FOREIGN_COLLECTORS)
        self.launch("--metrics-sock-name", self.metrics_sock)
        self.wait_for_line("Python engine started.")
        for _ in range(3):
            status, _, body = fetch(self.metrics_sock)
            self.assertEqual(status, 200)
            self.assertEqual(sample_names(body), {"vllm:djl_test_gauge"}, body)
        self.assertNotIn("render failed", "".join(self.log))

    def test_worker_serves_nothing_without_the_argument(self):
        self.launch()
        self.wait_for_line("Python engine started.")
        self.assertFalse(os.path.lexists(self.metrics_sock))
        self.assertNotIn("engine metrics", "".join(self.log))

    def test_worker_survives_a_metrics_socket_it_cannot_create(self):
        path = os.path.join(self.tmp, "missing", "m.metrics")
        self.launch("--metrics-sock-name", path)
        self.wait_for_line("Python engine started.")
        output = "".join(self.log)
        self.assertIn("engine metrics disabled", output)
        self.assertFalse(os.path.lexists(path))

    def test_worker_does_not_serve_metrics_under_mpi(self):
        self.launch("--metrics-sock-name",
                    self.metrics_sock,
                    env={"OMPI_COMM_WORLD_RANK": "1"})
        self.wait_for_line("Python engine started.")
        self.assertFalse(os.path.lexists(self.metrics_sock))
        self.assertNotIn("engine metrics", "".join(self.log))
        # the request socket carries the rank
        self.assertTrue(os.path.exists(self.sock + ".1"))


if __name__ == '__main__':
    unittest.main()
