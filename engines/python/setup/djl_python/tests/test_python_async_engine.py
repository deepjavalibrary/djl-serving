import queue
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.python_async_engine import PythonAsyncEngine, REQUEST_TRACKING_ID_KEY
from djl_python.python_sync_engine import PythonSyncEngine


async def fake_async_generator(items):
    """A real async generator (types.AsyncGeneratorType), since invoke_handler
    checks isinstance(outputs, types.AsyncGeneratorType) specifically - a
    class merely implementing __aiter__/__anext__ does not satisfy that
    check, only an object produced by an `async def ... yield` function does.
    """
    for item in items:
        yield item


def make_engine():
    # Run the real PythonAsyncEngine.__init__ (with PythonSyncEngine.__init__
    # stubbed out, since it opens a real socket and needs a full `args`
    # object) so output_queue/exception_queue/loop are whatever production
    # actually constructs. Bypassing __init__ entirely and setting
    # output_queue = Queue() ourselves would still pass even if production
    # reverted to asyncio.Queue.
    with patch.object(PythonSyncEngine, "__init__", return_value=None):
        engine = PythonAsyncEngine(MagicMock(), MagicMock())
    engine.service = MagicMock()
    engine.cl_socket = MagicMock()
    return engine


def make_input(tracking_id="req-1"):
    inputs = Input()
    inputs.properties[REQUEST_TRACKING_ID_KEY] = tracking_id
    return inputs


class TestPythonAsyncEngineOutputQueue(unittest.IsolatedAsyncioTestCase):
    """Regression coverage for the output_queue producer/consumer handoff in
    PythonAsyncEngine. This queue used to be an asyncio.Queue, requiring every
    dequeue in send_responses() to round-trip through asyncio.run_coroutine_threadsafe
    back onto the event loop - a cross-thread wakeup per streamed token that
    serialized token delivery for all concurrent requests through the loop
    for no reason, since only the blocking socket write actually needs to be
    off the loop. It is now a plain, thread-safe queue.Queue: invoke_handler
    (running on the event loop) calls put_nowait() directly with no await, and
    send_responses (running on its own OS thread) calls a plain blocking get()
    with no event-loop interaction at all.
    """

    async def test_invoke_handler_non_streaming_output_reaches_queue(self):
        engine = make_engine()
        output = Output(code=200, message="OK")
        engine.service.invoke_handler_async = MagicMock(
            return_value=_awaitable(output))

        await engine.invoke_handler("infer", make_input("req-1"))

        self.assertEqual(engine.output_queue.qsize(), 1)
        queued = engine.output_queue.get_nowait()
        self.assertIs(queued, output)
        self.assertEqual(queued.get_property(REQUEST_TRACKING_ID_KEY), "req-1")

    async def test_invoke_handler_streaming_outputs_all_reach_queue_in_order(
            self):
        engine = make_engine()
        chunks = [Output(message=f"chunk-{i}") for i in range(5)]
        engine.service.invoke_handler_async = MagicMock(
            return_value=_awaitable(fake_async_generator(chunks)))

        await engine.invoke_handler("infer", make_input("req-2"))

        self.assertEqual(engine.output_queue.qsize(), 5)
        for expected_chunk in chunks:
            queued = engine.output_queue.get_nowait()
            self.assertIs(queued, expected_chunk)
            self.assertEqual(queued.get_property(REQUEST_TRACKING_ID_KEY),
                             "req-2")

    async def test_invoke_handler_oom_error_queues_507(self):
        engine = make_engine()

        async def raise_oom(*_args, **_kwargs):
            raise MemoryError("CUDA error: out of memory")

        engine.service.invoke_handler_async = raise_oom

        await engine.invoke_handler("infer", make_input("req-3"))

        queued = engine.output_queue.get_nowait()
        self.assertEqual(queued.code, 507)

    async def test_invoke_handler_generic_error_queues_424(self):
        engine = make_engine()

        async def raise_generic(*_args, **_kwargs):
            raise ValueError("boom")

        engine.service.invoke_handler_async = raise_generic

        await engine.invoke_handler("infer", make_input("req-4"))

        queued = engine.output_queue.get_nowait()
        self.assertEqual(queued.code, 424)

    async def test_invoke_handler_none_output_queues_204(self):
        engine = make_engine()
        engine.service.invoke_handler_async = MagicMock(
            return_value=_awaitable(None))

        await engine.invoke_handler("infer", make_input("req-5"))

        queued = engine.output_queue.get_nowait()
        self.assertEqual(queued.code, 204)


class TestSendResponsesCrossThreadHandoff(unittest.TestCase):
    """Exercises the real thread boundary: outputs enqueued from one thread
    (simulating the event-loop thread) must be dequeued and sent by
    send_responses() running on a separate OS thread, with no asyncio loop
    involved on the consumer side at all - engine.loop is deliberately left
    None to prove send_responses no longer depends on it."""

    def test_send_responses_drains_queue_across_threads_without_event_loop(
            self):
        engine = make_engine()
        self.assertIsNone(engine.loop)

        n_items = 50
        outputs = [Output(message=str(i)) for i in range(n_items)]
        sent = []
        engine.cl_socket = object()  # sentinel, just needs identity
        for o in outputs:
            o.send = lambda sock, o=o: sent.append((o, sock))

        for o in outputs:
            engine.output_queue.put_nowait(o)

        # Run the real production method - not a reimplementation of its
        # loop - so a regression in send_responses() itself (e.g. reverting
        # to the asyncio.Queue/run_coroutine_threadsafe round trip) would
        # actually be caught here. It loops forever, so run it on a
        # background daemon thread and just wait for it to drain everything
        # enqueued above, rather than for it to return.
        t = threading.Thread(target=engine.send_responses, daemon=True)
        t.start()

        deadline = time.time() + 5
        while len(sent) < n_items and time.time() < deadline:
            time.sleep(0.01)

        self.assertEqual(len(sent), n_items)
        self.assertEqual([o for o, _ in sent], outputs)
        self.assertTrue(all(sock is engine.cl_socket for _, sock in sent))

    def test_output_queue_is_plain_thread_safe_queue_not_asyncio_queue(self):
        # Guards against regressing back to asyncio.Queue, which would
        # silently reintroduce the cross-thread round trip this fix removes:
        # asyncio.Queue.get()/put() are coroutines and are not safe to call
        # directly from a non-loop thread without run_coroutine_threadsafe.
        engine = make_engine()
        self.assertIsInstance(engine.output_queue, queue.Queue)
        self.assertNotIn("asyncio", type(engine.output_queue).__module__)


async def _awaitable(value):
    return value


if __name__ == "__main__":
    unittest.main()
