#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import asyncio
import logging
import time
import traceback
import types
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Thread
from queue import Queue

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.python_sync_engine import PythonSyncEngine

REQUEST_TRACKING_ID_KEY = "request_tracking_id"


class PythonAsyncEngine(PythonSyncEngine):
    """
    Backend engine to run python code in decoupled/async mode.
    Requests are forwarded from the model server and submitted to the handler.
    The handler returns responses as they become available, and sends them to the frontend.
    Requests are tracked/coordinated via the request_tracking_id property.
    This is an internal property set/managed by the model server.
    """

    def __init__(self, args, service):
        super().__init__(args, service)
        # A plain thread-safe queue.Queue, not asyncio.Queue: outputs are
        # produced on the event-loop thread (invoke_handler) and consumed on a
        # dedicated OS thread (send_responses). queue.Queue.put_nowait/get are
        # safe to call across threads with no event-loop involvement, unlike
        # asyncio.Queue, which requires scheduling every access back onto the
        # loop via run_coroutine_threadsafe - that added a cross-thread round
        # trip (and a wakeup of whichever coroutine loses the GIL race for it)
        # per streamed token, serializing all concurrent requests' token
        # delivery through the event loop for no reason: the queue itself
        # doesn't need the loop, only the blocking socket write does.
        self.output_queue = Queue()
        self.exception_queue = Queue()
        self.loop = None
        # Todo: for async mode we should maybe consider

    def receive_requests(self):
        logging.info("starting receive requests thread")
        while True:
            inputs, function_name = self._prepare_inputs()
            logging.debug(
                f"received new request with tracking_id {inputs.get_property(REQUEST_TRACKING_ID_KEY)}, submitting to handler"
            )
            asyncio.run_coroutine_threadsafe(
                self.invoke_handler(function_name, inputs), self.loop)

    async def invoke_handler(self, function_name: str, inputs: Input):
        request_tracking_id = inputs.get_property(REQUEST_TRACKING_ID_KEY)
        try:
            outputs = await self.service.invoke_handler_async(
                function_name, inputs)
        except Exception as e:
            logging.exception("Failed invoke service.invoke_handler_async()")
            if (type(e).__name__ == "OutOfMemoryError"
                    or type(e).__name__ == "MemoryError"
                    or "No available memory for the cache blocks" in str(e)
                    or "CUDA error: out of memory" in str(e)):
                logging.exception(
                    f"Memory Error encountered when invoking module {self.service.module}, function {function_name}"
                )
                outputs = Output(code=507, message=str(e))
            else:
                logging.exception(
                    f"service.invoke_handler_async() failure. There was an error invoking module {self.service.module}, function {function_name}"
                )
                outputs = Output().error(
                    str(e), message="service.invoke_handler_async() failure")
        if outputs is None:
            outputs = Output(code=204, message="No content")
            logging.debug(
                "empty response received from service.invoke_handler_async()")
        elif not isinstance(outputs, Output) and not isinstance(
                outputs, types.AsyncGeneratorType):
            message = (
                f"Invalid type returned from {self.service.module}.{function_name}. "
                f"Received type {type(outputs)}, does not match expected type djl_python.outputs.Output or types.AsyncGenerator"
            )
            logging.error(message)
            outputs = Output().error(message)

        if isinstance(outputs, types.AsyncGeneratorType):
            async for output in outputs:
                output.add_property(REQUEST_TRACKING_ID_KEY,
                                    request_tracking_id)
                self.output_queue.put_nowait(output)
            return
        # Request tracking ID is needed always for async
        # Do this here so that users in custom handlers do not need to worry about it
        outputs.add_property(REQUEST_TRACKING_ID_KEY, request_tracking_id)
        logging.debug(f"putting result of inference to output queue")
        self.output_queue.put_nowait(outputs)

    def send_responses(self):
        logging.info("starting send responses thread")
        while True:
            logging.debug("waiting for new inference response")
            output = self.output_queue.get()
            output.send(self.cl_socket)

    def run_server(self):

        async def main():
            self.loop = asyncio.get_running_loop()
            self._create_cl_socket()

            def catch_all(func):
                try:
                    func()
                except Exception as e:
                    logging.error(f"{func} failed. Details {e}")
                    self.exception_queue.put(str(traceback.format_exc()))

            threads = [
                Thread(target=partial(catch_all, self.receive_requests)),
                Thread(target=partial(catch_all, self.send_responses)),
            ]

            for thread in threads:
                thread.start()

            def check_threads():
                while True:
                    if not all(t.is_alive() for t in threads):
                        return
                    time.sleep(1)

            with ThreadPoolExecutor(1) as executor:
                await asyncio.get_event_loop().run_in_executor(
                    executor, check_threads)

        try:
            import uvloop
            uvloop.install()
        except ImportError:
            logging.warning("uvloop not available, using asyncio as default")

        asyncio.run(main())

        if not self.exception_queue.empty():
            logging.error(
                f"djl async engine terminated with error {self.exception_queue.get()}"
            )
        logging.info("djl async engine terminated")
