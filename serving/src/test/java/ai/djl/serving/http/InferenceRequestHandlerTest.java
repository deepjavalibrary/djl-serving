/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.serving.http;

import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertTrue;

import java.lang.reflect.Field;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.TimeUnit;

import org.testng.annotations.Test;

/**
 * Regression coverage for {@code InferenceRequestHandler}'s streaming-response delivery
 * executor. Every streaming (or chunked non-streaming) response blocks its delivery thread on
 * {@code ChunkedBytesSupplier.nextChunk(...)} for the entire generation duration, so the executor
 * used to run that loop directly bounds how many concurrent responses the server can actually
 * deliver. The no-argument {@code CompletableFuture.whenCompleteAsync} defaults to {@code
 * ForkJoinPool.commonPool()}, whose parallelism is {@code availableProcessors() - 1}; on a small
 * host this becomes a hard concurrency ceiling regardless of backend capacity. These tests assert
 * the actual behavior that matters -- that the dedicated executor can run more concurrent blocking
 * tasks than the common pool's parallelism -- rather than asserting an implementation detail (e.g.
 * the concrete executor class), so the test still passes if the implementation is swapped for
 * another unbounded/adequately-sized executor.
 */
public class InferenceRequestHandlerTest {

    @Test
    public void testResponseExecutorIsNotForkJoinCommonPool() throws Exception {
        ExecutorService executor = getResponseExecutor();
        assertFalse(
                isForkJoinCommonPool(executor),
                "streaming response delivery must not use ForkJoinPool.commonPool(), whose"
                        + " parallelism is capped at availableProcessors() - 1 and is shared with"
                        + " unrelated JVM-wide async work");
    }

    @Test
    public void testResponseExecutorSupportsMoreConcurrentBlockingTasksThanCommonPoolParallelism()
            throws Exception {
        ExecutorService executor = getResponseExecutor();
        int commonPoolParallelism = ForkJoinPool.getCommonPoolParallelism();
        // Simulate what sendOutput's blocking nextChunk() loop does: many concurrent
        // streaming responses, each occupying a thread that blocks until released. If this
        // still ran on a pool bounded at commonPoolParallelism, only that many of these
        // tasks could be running at once and the rest would starve until one released its
        // thread -- exactly the p99 ceiling this fix removes.
        int concurrentStreams = commonPoolParallelism + 4;
        CountDownLatch allRunning = new CountDownLatch(concurrentStreams);
        CountDownLatch release = new CountDownLatch(1);

        for (int i = 0; i < concurrentStreams; ++i) {
            executor.submit(
                    () -> {
                        allRunning.countDown();
                        try {
                            release.await();
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                        }
                    });
        }

        boolean allStartedPromptly = allRunning.await(5, TimeUnit.SECONDS);
        release.countDown();

        assertTrue(
                allStartedPromptly,
                "expected all "
                        + concurrentStreams
                        + " concurrent blocking tasks to start promptly (more than the common"
                        + " pool's parallelism of "
                        + commonPoolParallelism
                        + "), but the executor did not have enough capacity");
    }

    @Test
    public void testResponseExecutorUsesDaemonThreads() throws Exception {
        ExecutorService executor = getResponseExecutor();
        CountDownLatch started = new CountDownLatch(1);
        boolean[] isDaemon = new boolean[1];

        executor.submit(
                () -> {
                    isDaemon[0] = Thread.currentThread().isDaemon();
                    started.countDown();
                });

        assertTrue(started.await(5, TimeUnit.SECONDS));
        assertTrue(
                isDaemon[0],
                "response-delivery threads must be daemon threads so they never block JVM"
                        + " shutdown");
    }

    private static ExecutorService getResponseExecutor() throws Exception {
        Field field = InferenceRequestHandler.class.getDeclaredField("RESPONSE_EXECUTOR");
        field.setAccessible(true);
        return (ExecutorService) field.get(null);
    }

    private static boolean isForkJoinCommonPool(ExecutorService executor) {
        return executor == ForkJoinPool.commonPool();
    }
}
