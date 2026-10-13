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

import io.netty.handler.codec.PrematureChannelClosureException;
import io.netty.handler.codec.TooLongFrameException;
import io.netty.handler.codec.http.HttpResponseStatus;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.net.ConnectException;
import java.nio.channels.ClosedChannelException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public class EngineMetricsClientTest {

    private static final int LIMIT = 1 << 20;
    private static final String BODY =
            "# HELP vllm:foo Foo.\n# TYPE vllm:foo counter\nvllm:foo{a=\"b\"} 1.0\n";

    @BeforeClass
    public void checkSupported() {
        if (!UnixSocketTestServer.isSupported()) {
            throw new SkipException("Unix domain sockets need Epoll or KQueue");
        }
    }

    @Test
    public void testOk() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(BODY))) {
            String uri = "/metrics?worker=gpu0&name%5B%5D=vllm%3Afoo";
            String body =
                    EngineMetricsClient.get(server.path(), uri, LIMIT, 2000)
                            .get(5, TimeUnit.SECONDS);
            Assert.assertEquals(body, BODY);
            Assert.assertEquals(
                    server.requests(),
                    Collections.singletonList("GET " + uri + " | localhost | close"));
        }
    }

    @Test
    public void testNonAscii() throws Exception {
        String text = "# HELP vllm:foo café モデル\nvllm:foo{m=\"ü\"} 1\n";
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(text))) {
            Assert.assertEquals(
                    EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                            .get(5, TimeUnit.SECONDS),
                    text);
        }
    }

    @Test
    public void testEmptyBody() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(""))) {
            Assert.assertEquals(
                    EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                            .get(5, TimeUnit.SECONDS),
                    "");
        }
    }

    @Test
    public void testNotFound() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(
                        UnixSocketTestServer.status(HttpResponseStatus.NOT_FOUND, "no"))) {
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/other", LIMIT, 2000));
            Assert.assertTrue(
                    error instanceof EngineMetricsClient.HttpStatusException, error.toString());
            Assert.assertEquals(((EngineMetricsClient.HttpStatusException) error).getStatus(), 404);
            Assert.assertEquals(EngineMetricsClient.describe(error), "HTTP status 404");
        }
    }

    @Test
    public void testServerError() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(
                        UnixSocketTestServer.status(
                                HttpResponseStatus.INTERNAL_SERVER_ERROR, "x"))) {
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000));
            Assert.assertEquals(EngineMetricsClient.describe(error), "HTTP status 500");
        }
    }

    @Test
    public void testSlowPastDeadline() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.silent())) {
            long begin = System.nanoTime();
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 300));
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            Assert.assertTrue(error instanceof TimeoutException, error.toString());
            Assert.assertEquals(EngineMetricsClient.describe(error), "timeout");
            Assert.assertTrue(elapsed >= 250 && elapsed < 2500, "elapsed " + elapsed);
            Assert.assertEquals(server.requests().size(), 1);
        }
    }

    @Test
    public void testSlowWithinDeadline() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.delayed(150, BODY))) {
            String body =
                    EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                            .get(5, TimeUnit.SECONDS);
            Assert.assertEquals(body, BODY);
        }
    }

    @Test
    public void testRefused() throws Exception {
        // A file that is not a socket behaves like a stale socket file: nothing accepts on it.
        Path dir = Files.createTempDirectory("djlem");
        Path stale = Files.createFile(dir.resolve("stale.metrics"));
        try {
            long begin = System.nanoTime();
            Throwable error =
                    failure(EngineMetricsClient.get(stale.toString(), "/metrics", LIMIT, 10000));
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            Assert.assertTrue(error instanceof ConnectException, error.toString());
            Assert.assertTrue(
                    EngineMetricsClient.describe(error).startsWith("cannot connect"),
                    EngineMetricsClient.describe(error));
            Assert.assertTrue(elapsed < 5000, "failed only at the deadline: " + elapsed);
        } finally {
            Files.deleteIfExists(stale);
            Files.deleteIfExists(dir);
        }
    }

    @Test
    public void testMissingSocketFile() throws Exception {
        Path dir = Files.createTempDirectory("djlem");
        try {
            String missing = dir.resolve("none.metrics").toString();
            Throwable error = failure(EngineMetricsClient.get(missing, "/metrics", LIMIT, 2000));
            Assert.assertTrue(error instanceof IOException, error.toString());
            Assert.assertEquals(EngineMetricsClient.describe(error), "socket file missing");
        } finally {
            Files.deleteIfExists(dir);
        }
    }

    @Test
    public void testClosedEarly() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.closing())) {
            long begin = System.nanoTime();
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 5000));
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            Assert.assertTrue(error instanceof ClosedChannelException, error.toString());
            Assert.assertEquals(EngineMetricsClient.describe(error), "closed before the response");
            Assert.assertTrue(elapsed < 2500, "failed only at the deadline: " + elapsed);
        }
    }

    @Test
    public void testOversize() throws Exception {
        String big = "x".repeat(64 * 1024);
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(big))) {
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", 1024, 2000));
            Assert.assertTrue(rootCause(error) instanceof TooLongFrameException, error.toString());
            Assert.assertEquals(
                    EngineMetricsClient.describe(error), "response over the size limit");
        }
    }

    @Test
    public void testBodyAtTheLimit() throws Exception {
        String exact = "y".repeat(1024);
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(exact))) {
            Assert.assertEquals(
                    EngineMetricsClient.get(server.path(), "/metrics", 1024, 2000)
                            .get(5, TimeUnit.SECONDS),
                    exact);
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", 1023, 2000));
            Assert.assertTrue(rootCause(error) instanceof TooLongFrameException, error.toString());
        }
    }

    @Test
    public void testBodyEndedByClose() throws Exception {
        // HTTP/1.0 without Content-Length: the body ends when the peer closes
        String raw = "HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nvllm:foo 1\n";
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.raw(raw))) {
            Assert.assertEquals(
                    EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                            .get(5, TimeUnit.SECONDS),
                    "vllm:foo 1\n");
        }
    }

    @Test
    public void testTruncatedBody() throws Exception {
        String raw = "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n0123456789";
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.raw(raw))) {
            long begin = System.nanoTime();
            Throwable error =
                    failure(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 5000));
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            Assert.assertTrue(error instanceof PrematureChannelClosureException, error.toString());
            Assert.assertEquals(EngineMetricsClient.describe(error), "closed before the response");
            Assert.assertTrue(elapsed < 2500, "failed only at the deadline: " + elapsed);
        }
    }

    @Test
    public void testMalformedResponses() throws Exception {
        String chunk =
                "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nabc\r\n0\r\n\r\n";
        String garbage = "not http at all\r\n\r\n";
        for (String raw : new String[] {chunk, garbage}) {
            try (UnixSocketTestServer server =
                    UnixSocketTestServer.start(UnixSocketTestServer.raw(raw))) {
                Throwable error =
                        failure(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 5000));
                Assert.assertTrue(error instanceof IOException, error.toString());
                Assert.assertEquals(
                        EngineMetricsClient.describe(error),
                        "IOException: Malformed HTTP response");
            }
        }
    }

    @Test
    public void testManyRequests() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.ok(BODY))) {
            for (int i = 0; i < 50; ++i) {
                Assert.assertEquals(
                        EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                                .get(5, TimeUnit.SECONDS),
                        BODY);
            }
            List<CompletableFuture<String>> pending = new ArrayList<>();
            for (int i = 0; i < 16; ++i) {
                pending.add(EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 5000));
            }
            for (CompletableFuture<String> future : pending) {
                Assert.assertEquals(future.get(10, TimeUnit.SECONDS), BODY);
            }
            Assert.assertEquals(server.requests().size(), 66);
            Assert.assertEquals(server.connections(), 66);
        }
    }

    @Test
    public void testCompletesOnTheClientThread() throws Exception {
        try (UnixSocketTestServer server =
                UnixSocketTestServer.start(UnixSocketTestServer.delayed(300, BODY))) {
            String thread =
                    EngineMetricsClient.get(server.path(), "/metrics", LIMIT, 2000)
                            .thenApply(body -> Thread.currentThread().getName())
                            .get(5, TimeUnit.SECONDS);
            Assert.assertTrue(thread.startsWith("djl-engine-metrics"), thread);
        }
    }

    @Test
    public void testDescribe() {
        Assert.assertEquals(EngineMetricsClient.describe(new TimeoutException("t")), "timeout");
        Assert.assertEquals(
                EngineMetricsClient.describe(new ExecutionException(new NoSuchFileException("/x"))),
                "socket file missing");
        Assert.assertEquals(
                EngineMetricsClient.describe(new IllegalStateException("boom")),
                "IllegalStateException: boom");
        Assert.assertEquals(
                EngineMetricsClient.describe(new IllegalStateException()), "IllegalStateException");
    }

    private static Throwable failure(CompletableFuture<String> future) throws InterruptedException {
        try {
            String body = future.get(10, TimeUnit.SECONDS);
            throw new AssertionError("Expected a failure, got " + body.length() + " characters");
        } catch (ExecutionException e) {
            return e.getCause();
        } catch (TimeoutException e) {
            throw new AssertionError("The future did not complete", e);
        }
    }

    private static Throwable rootCause(Throwable error) {
        Throwable cause = error;
        while (cause.getCause() != null) {
            cause = cause.getCause();
        }
        return cause;
    }
}
