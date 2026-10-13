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

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import ai.djl.Device;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.http.EngineMetrics.Result;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.Workflow;

import io.netty.handler.codec.http.HttpResponseStatus;

import org.slf4j.Logger;
import org.slf4j.helpers.MessageFormatter;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

public class EngineMetricsTest {

    private static final long SECOND = TimeUnit.SECONDS.toNanos(1);

    private static final String W0 =
            "# HELP vllm:requests_total Requests.\n"
                    + "# TYPE vllm:requests_total counter\n"
                    + "vllm:requests_total{model_name=\"m\",worker=\"gpu0\"} 3.0\n"
                    + "# HELP vllm:running Running.\n"
                    + "# TYPE vllm:running gauge\n"
                    + "vllm:running{model_name=\"m\",worker=\"gpu0\"} 1.0\n";

    private static final String W1 =
            "# HELP vllm:requests_total Requests.\n"
                    + "# TYPE vllm:requests_total counter\n"
                    + "vllm:requests_total{model_name=\"m\",worker=\"gpu1\"} 5.0\n"
                    + "# HELP vllm:running Running.\n"
                    + "# TYPE vllm:running gauge\n"
                    + "vllm:running{model_name=\"m\",worker=\"gpu1\"} 2.0\n";

    private static final String MERGED =
            "# HELP vllm:requests_total Requests.\n"
                    + "# TYPE vllm:requests_total counter\n"
                    + "vllm:requests_total{model_name=\"m\",worker=\"gpu0\"} 3.0\n"
                    + "vllm:requests_total{model_name=\"m\",worker=\"gpu1\"} 5.0\n"
                    + "# HELP vllm:running Running.\n"
                    + "# TYPE vllm:running gauge\n"
                    + "vllm:running{model_name=\"m\",worker=\"gpu0\"} 1.0\n"
                    + "vllm:running{model_name=\"m\",worker=\"gpu1\"} 2.0\n";

    @Test
    public void testIsEnabled() {
        String name = "SERVING_ENGINE_METRICS";
        if (System.getenv(name) != null) {
            throw new SkipException(name + " is set in the environment");
        }
        String old = System.getProperty(name);
        try {
            System.clearProperty(name);
            Assert.assertFalse(EngineMetrics.isEnabled());
            System.setProperty(name, "true");
            Assert.assertTrue(EngineMetrics.isEnabled());
            System.setProperty(name, "TRUE");
            Assert.assertTrue(EngineMetrics.isEnabled());
            System.setProperty(name, "false");
            Assert.assertFalse(EngineMetrics.isEnabled());
            System.setProperty(name, "1");
            Assert.assertFalse(EngineMetrics.isEnabled());
        } finally {
            restore(name, old);
        }
    }

    @Test
    public void testLimits() {
        Assert.assertEquals(EngineMetrics.MAX_BODY_BYTES, 4 * 1024 * 1024);
        Assert.assertEquals(EngineMetrics.TIMEOUT_MS, 2000);
        Assert.assertEquals(EngineMetrics.REUSE_NANOS, TimeUnit.SECONDS.toNanos(1));
        Assert.assertEquals(EngineMetrics.LOG_INTERVAL_NANOS, TimeUnit.MINUTES.toNanos(5));
        Assert.assertEquals(EngineMetrics.STUCK_NANOS, TimeUnit.SECONDS.toNanos(8));
        Assert.assertEquals(EngineMetrics.KEY_PREFIX, "engine_metrics_socket.");
        Assert.assertEquals(EngineMetrics.SOCKET_SUFFIX, ".metrics");
    }

    @Test
    public void testMergeZeroOrOneBodyIsUnchanged() {
        Assert.assertEquals(EngineMetrics.merge(Collections.emptyList()), "");
        String odd = "no header 1\n\n# stray comment\n# HELP x y";
        Assert.assertSame(EngineMetrics.merge(Collections.singletonList(odd)), odd);
    }

    @Test
    public void testMergeGroupsByFamily() {
        Assert.assertEquals(EngineMetrics.merge(Arrays.asList(W0, W1)), MERGED);
        // the order of the bodies is the order of the samples
        String reversed =
                "# HELP vllm:requests_total Requests.\n"
                        + "# TYPE vllm:requests_total counter\n"
                        + "vllm:requests_total{model_name=\"m\",worker=\"gpu1\"} 5.0\n"
                        + "vllm:requests_total{model_name=\"m\",worker=\"gpu0\"} 3.0\n"
                        + "# HELP vllm:running Running.\n"
                        + "# TYPE vllm:running gauge\n"
                        + "vllm:running{model_name=\"m\",worker=\"gpu1\"} 2.0\n"
                        + "vllm:running{model_name=\"m\",worker=\"gpu0\"} 1.0\n";
        Assert.assertEquals(EngineMetrics.merge(Arrays.asList(W1, W0)), reversed);
    }

    @Test
    public void testMergeKeepsFirstSeenFamilyOrder() {
        String first = "# HELP b B.\n# TYPE b gauge\nb 1\n# HELP a A.\n# TYPE a gauge\na 1\n";
        String second = "# HELP c C.\n# TYPE c gauge\nc 2\n# HELP a A.\n# TYPE a gauge\na 2\n";
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList(first, second)),
                "# HELP b B.\n# TYPE b gauge\nb 1\n"
                        + "# HELP a A.\n# TYPE a gauge\na 1\na 2\n"
                        + "# HELP c C.\n# TYPE c gauge\nc 2\n");
    }

    @Test
    public void testMergeKeepsHeadersOfTheFirstSource() {
        String first = "# HELP a First help.\n# TYPE a counter\na 1\n";
        String second = "# HELP a Other help.\n# TYPE a gauge\na 2\n";
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList(first, second)),
                "# HELP a First help.\n# TYPE a counter\na 1\na 2\n");
        // a header that the first source lacks is taken from the next one that has it
        String bare = "# TYPE a counter\na 1\n";
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList(bare, second)),
                "# TYPE a counter\n# HELP a Other help.\na 1\na 2\n");
    }

    @Test
    public void testMergeDropsOtherCommentsAndBlankLines() {
        String first = "# HELP a A.\n# TYPE a gauge\n\na 1\n# UNIT a seconds\n# EOF\n";
        String second = "  \n# HELP a A.\r\n# TYPE a gauge\r\n# stray\r\na 2\r\n";
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList(first, second)),
                "# HELP a A.\n# TYPE a gauge\na 1\na 2\n");
    }

    @Test
    public void testMergeHandlesMissingFinalNewline() {
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList("# TYPE a gauge\na 1", "# TYPE a gauge\na 2")),
                "# TYPE a gauge\na 1\na 2\n");
    }

    @Test
    public void testMergeKeepsSamplesThatPrecedeAnyHeader() {
        String first = "orphan_one 1\n# HELP a A.\n# TYPE a gauge\na 1\n";
        String second = "orphan_two 2\n# HELP a A.\n# TYPE a gauge\na 2\n";
        // the headerless group comes first, and a body's leading samples never join the last
        // family of the previous body
        Assert.assertEquals(
                EngineMetrics.merge(Arrays.asList(first, second)),
                "orphan_one 1\norphan_two 2\n# HELP a A.\n# TYPE a gauge\na 1\na 2\n");
    }

    @Test
    public void testMergeKeepsHistogramSamplesTogether() {
        String hist0 =
                "# HELP h H.\n# TYPE h histogram\n"
                        + "h_bucket{le=\"1.0\",worker=\"gpu0\"} 1\n"
                        + "h_bucket{le=\"+Inf\",worker=\"gpu0\"} 2\n"
                        + "h_count{worker=\"gpu0\"} 2\n"
                        + "h_sum{worker=\"gpu0\"} 3\n";
        String hist1 = hist0.replace("gpu0", "gpu1");
        String merged = EngineMetrics.merge(Arrays.asList(hist0, hist1));
        Assert.assertEquals(
                merged,
                "# HELP h H.\n# TYPE h histogram\n"
                        + "h_bucket{le=\"1.0\",worker=\"gpu0\"} 1\n"
                        + "h_bucket{le=\"+Inf\",worker=\"gpu0\"} 2\n"
                        + "h_count{worker=\"gpu0\"} 2\n"
                        + "h_sum{worker=\"gpu0\"} 3\n"
                        + "h_bucket{le=\"1.0\",worker=\"gpu1\"} 1\n"
                        + "h_bucket{le=\"+Inf\",worker=\"gpu1\"} 2\n"
                        + "h_count{worker=\"gpu1\"} 2\n"
                        + "h_sum{worker=\"gpu1\"} 3\n");
    }

    @Test
    public void testLabelRule() {
        // one source and a single-model container: no label
        Harness single = new Harness(sources("gpu0"), false);
        single.fetcher.bodies.put(path("gpu0"), W0);
        Result result = single.scrape();
        Assert.assertEquals(
                single.fetcher.calls, Collections.singletonList(path("gpu0") + " /metrics"));
        Assert.assertEquals(result.body, W0);

        // two sources: label
        Harness two = new Harness(sources("gpu0", "gpu1"), false);
        two.fetcher.bodies.put(path("gpu0"), W0);
        two.fetcher.bodies.put(path("gpu1"), W1);
        two.scrape();
        Assert.assertEqualsNoOrder(
                two.fetcher.calls.toArray(),
                new Object[] {
                    path("gpu0") + " /metrics?worker=gpu0", path("gpu1") + " /metrics?worker=gpu1"
                });

        // one source in a multi-model container: label
        Harness multi = new Harness(sources("gpu0"), true);
        multi.fetcher.bodies.put(path("gpu0"), W0);
        multi.scrape();
        Assert.assertEquals(
                multi.fetcher.calls,
                Collections.singletonList(path("gpu0") + " /metrics?worker=gpu0"));
    }

    @Test
    public void testLabelRuleDoesNotDependOnFetchSuccess() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        h.scrape();
        h.fetcher.calls.clear();

        // the second worker stops answering: the survivor is still asked for its label
        h.fetcher.failing.add(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        Result result = h.scrape();
        Assert.assertEquals(result.registered, 2);
        Assert.assertEquals(result.ok, 1);
        Assert.assertEquals(result.body, W0);
        Assert.assertTrue(h.fetcher.calls.contains(path("gpu0") + " /metrics?worker=gpu0"));

        // the rule follows the registrations
        h.sources.remove("gpu1");
        h.fetcher.calls.clear();
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.fetcher.calls, Collections.singletonList(path("gpu0") + " /metrics"));
    }

    @Test
    public void testStatusInputs() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        Result all = h.scrape();
        Assert.assertEquals(all.registered, 2);
        Assert.assertEquals(all.ok, 2);
        Assert.assertEquals(all.body, MERGED);

        // one failed: the survivor's text is returned as it came
        h.fetcher.failing.add(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        Result some = h.scrape();
        Assert.assertEquals(some.registered, 2);
        Assert.assertEquals(some.ok, 1);
        Assert.assertEquals(some.body, W0);

        // every one failed
        h.fetcher.failing.add(path("gpu0"));
        h.clock.addAndGet(10 * SECOND);
        Result none = h.scrape();
        Assert.assertEquals(none.registered, 2);
        Assert.assertEquals(none.ok, 0);
        Assert.assertEquals(none.body, "");

        // nothing registered
        h.sources.clear();
        h.clock.addAndGet(10 * SECOND);
        Result empty = h.scrape();
        Assert.assertEquals(empty.registered, 0);
        Assert.assertEquals(empty.ok, 0);
        Assert.assertEquals(empty.body, "");
    }

    @Test
    public void testFetcherThatThrowsIsAFailure() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu1"), W1);
        h.fetcher.throwing.add(path("gpu0"));
        Result result = h.scrape();
        Assert.assertEquals(result.registered, 2);
        Assert.assertEquals(result.ok, 1);
        Assert.assertEquals(result.body, W1);
    }

    @Test
    public void testFlightOfAFanOutThatThrowsAnErrorCompletes() throws Exception {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.fault = new OutOfMemoryError("unable to create native thread");
        try {
            h.metrics.scrapeSources(Collections.emptyList());
            Assert.fail("the error reaches the caller");
        } catch (OutOfMemoryError expected) {
            // expected
        }

        // a scrape with the same key right after fails at once, it does not wait for ever
        CompletableFuture<Result> joined = h.metrics.scrapeSources(Collections.emptyList());
        try {
            joined.get(10, TimeUnit.SECONDS);
            Assert.fail("the flight of the failed fan-out has no result");
        } catch (ExecutionException e) {
            Assert.assertTrue(e.getCause() instanceof OutOfMemoryError, e.toString());
        } catch (TimeoutException e) {
            throw new AssertionError("the flight of the failed fan-out never completes", e);
        }
        Assert.assertEquals(h.fetcher.calls.size(), 1);

        // after the reuse window the next scrape fetches again
        h.clock.addAndGet(SECOND);
        Result result = h.scrape();
        Assert.assertEquals(result.ok, 1);
        Assert.assertEquals(result.body, W0);
        Assert.assertEquals(h.fetcher.calls.size(), 2);
    }

    @Test
    public void testFlightThatLostItsCompletionIsReplaced() throws Exception {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        CompletableFuture<Result> first = h.metrics.scrapeSources(Collections.emptyList());
        Assert.assertEquals(h.fetcher.calls.size(), 1);

        // as long as the client can still answer, the flight is shared
        h.clock.addAndGet(EngineMetrics.STUCK_NANOS - 1);
        h.metrics.scrapeSources(Collections.emptyList());
        Assert.assertEquals(h.fetcher.calls.size(), 1);

        // long after every deadline of the client it is not
        h.clock.addAndGet(1);
        CompletableFuture<Result> second = h.metrics.scrapeSources(Collections.emptyList());
        Assert.assertEquals(h.fetcher.calls.size(), 2, "the stuck flight is still shared");
        Assert.assertFalse(second.isDone());

        // the answers of both fetches are used by the callers that wait for them
        h.fetcher.complete(path("gpu0"), W0);
        Assert.assertEquals(second.get(10, TimeUnit.SECONDS).body, W0);
        Assert.assertEquals(first.get(10, TimeUnit.SECONDS).body, W0);
    }

    @Test
    public void testUnfinishedFetchesAreCapped() throws Exception {
        int max = EngineMetrics.MAX_UNFINISHED;
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.hold = true;
        List<CompletableFuture<Result>> started = new ArrayList<>();
        for (int i = 0; i < max; ++i) {
            started.add(h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + i)));
        }
        // one fetch per source for each of them
        Assert.assertEquals(h.fetcher.calls.size(), 2 * max);

        // one more with other filters is refused at once and fetches nothing
        CompletableFuture<Result> refused =
                h.metrics.scrapeSources(Collections.singletonList("vllm:other"));
        Assert.assertTrue(refused.isCompletedExceptionally());
        try {
            refused.get(10, TimeUnit.SECONDS);
            Assert.fail("the scrape is refused");
        } catch (ExecutionException e) {
            Assert.assertTrue(e.getCause() instanceof ServiceUnavailableException, e.toString());
            Assert.assertEquals(
                    e.getCause().getMessage(),
                    "Engine metrics are busy: "
                            + max
                            + " fetches are unfinished, try again later.");
        }
        Assert.assertEquals(h.fetcher.calls.size(), 2 * max);

        // a scrape that shares the latest fetch does not start one, so it is not refused
        CompletableFuture<Result> shared =
                h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + (max - 1)));
        Assert.assertFalse(shared.isCompletedExceptionally());
        Assert.assertEquals(h.fetcher.calls.size(), 2 * max);

        // the answers arrive: every fetch is used by the scrapes that wait for it
        h.fetcher.complete(path("gpu0"), W0);
        h.fetcher.complete(path("gpu1"), W1);
        for (CompletableFuture<Result> result : started) {
            Assert.assertEquals(result.get(10, TimeUnit.SECONDS).ok, 2);
        }
        Assert.assertEquals(shared.get(10, TimeUnit.SECONDS).body, MERGED);

        // all of them are finished: there is room again
        h.fetcher.hold = false;
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        Result next = h.scrape(Collections.singletonList("vllm:other"));
        Assert.assertEquals(next.ok, 2);
        Assert.assertEquals(h.fetcher.calls.size(), 2 * max + 2);
    }

    @Test
    public void testFinishedFetchMakesRoomForOneMore() throws Exception {
        int max = EngineMetrics.MAX_UNFINISHED;
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        for (int i = 0; i < max; ++i) {
            h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + i));
        }
        Assert.assertTrue(
                h.metrics
                        .scrapeSources(Collections.singletonList("vllm:a"))
                        .isCompletedExceptionally());

        // the first fetch finishes, the other ones do not
        h.fetcher.held.get(path("gpu0")).get(0).complete(W0);
        CompletableFuture<Result> admitted =
                h.metrics.scrapeSources(Collections.singletonList("vllm:b"));
        Assert.assertFalse(admitted.isCompletedExceptionally());
        Assert.assertEquals(h.fetcher.calls.size(), max + 1);
        Assert.assertTrue(
                h.metrics
                        .scrapeSources(Collections.singletonList("vllm:c"))
                        .isCompletedExceptionally());
        Assert.assertEquals(h.fetcher.calls.size(), max + 1);
    }

    @Test
    public void testFetchThatIsLostDoesNotHoldASlot() {
        int max = EngineMetrics.MAX_UNFINISHED;
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        for (int i = 0; i < max; ++i) {
            h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + i));
        }

        // the client can still answer for them
        h.clock.addAndGet(EngineMetrics.STUCK_NANOS - 1);
        Assert.assertTrue(
                h.metrics
                        .scrapeSources(Collections.singletonList("vllm:late"))
                        .isCompletedExceptionally());

        // long after every deadline of the client they are lost, and a lost fetch is not counted
        h.clock.addAndGet(1);
        CompletableFuture<Result> admitted =
                h.metrics.scrapeSources(Collections.singletonList("vllm:late"));
        Assert.assertFalse(admitted.isCompletedExceptionally());
        Assert.assertEquals(h.fetcher.calls.size(), max + 1);
        for (int i = 0; i < max - 1; ++i) {
            Assert.assertFalse(
                    h.metrics
                            .scrapeSources(Collections.singletonList("vllm:again_" + i))
                            .isCompletedExceptionally());
        }
        Assert.assertTrue(
                h.metrics
                        .scrapeSources(Collections.singletonList("vllm:again"))
                        .isCompletedExceptionally());
    }

    @Test
    public void testConcurrentScrapesWithDifferentFiltersStayWithinTheCap() throws Exception {
        int max = EngineMetrics.MAX_UNFINISHED;
        int callers = 200;
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        ExecutorService pool = Executors.newFixedThreadPool(callers);
        try {
            CountDownLatch ready = new CountDownLatch(callers);
            CountDownLatch go = new CountDownLatch(1);
            List<Future<CompletableFuture<Result>>> calls = new ArrayList<>();
            for (int i = 0; i < callers; ++i) {
                List<String> names = Collections.singletonList("vllm:name_" + i);
                calls.add(
                        pool.submit(
                                () -> {
                                    ready.countDown();
                                    go.await();
                                    return h.metrics.scrapeSources(names);
                                }));
            }
            Assert.assertTrue(ready.await(10, TimeUnit.SECONDS));
            go.countDown();
            int refused = 0;
            for (Future<CompletableFuture<Result>> call : calls) {
                if (call.get(10, TimeUnit.SECONDS).isCompletedExceptionally()) {
                    ++refused;
                }
            }
            Assert.assertEquals(refused, callers - max);
            Assert.assertEquals(h.fetcher.calls.size(), max);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void testRefusedScrapesAreLoggedOncePerInterval() {
        int max = EngineMetrics.MAX_UNFINISHED;
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        int distinct = 0;
        for (int i = 0; i < max; ++i) {
            h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + distinct++));
        }
        Assert.assertTrue(h.logs.isEmpty(), h.logs.toString());

        // the first refused scrape is reported, the next ones only counted
        for (int i = 0; i < 3; ++i) {
            Assert.assertTrue(
                    h.metrics
                            .scrapeSources(Collections.singletonList("vllm:name_" + distinct++))
                            .isCompletedExceptionally());
        }
        String message =
                "WARN Engine metrics: "
                        + max
                        + " fetches are unfinished, refusing scrapes that need another one ("
                        + "%d refused since the last message)";
        Assert.assertEquals(h.logs, Collections.singletonList(String.format(message, 1)));

        // after the interval the fetches are lost and others take their place
        h.clock.addAndGet(EngineMetrics.LOG_INTERVAL_NANOS);
        for (int i = 0; i < max; ++i) {
            h.metrics.scrapeSources(Collections.singletonList("vllm:name_" + distinct++));
        }
        Assert.assertEquals(h.logs.size(), 1, h.logs.toString());
        Assert.assertTrue(
                h.metrics
                        .scrapeSources(Collections.singletonList("vllm:name_" + distinct++))
                        .isCompletedExceptionally());
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());
        Assert.assertEquals(h.logs.get(1), String.format(message, 3));
    }

    @Test
    public void testSourceSupplierFailure() {
        EngineMetrics metrics =
                new EngineMetrics(
                        () -> {
                            throw new IllegalStateException("broken");
                        },
                        () -> false,
                        (socket, uri) -> CompletableFuture.completedFuture(""),
                        System::nanoTime,
                        mock(Logger.class));
        CompletableFuture<Result> future = metrics.scrapeSources(Collections.emptyList());
        Assert.assertTrue(future.isCompletedExceptionally());
    }

    @Test
    public void testSingleFlight() throws Exception {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.hold = true;
        ExecutorService pool = Executors.newFixedThreadPool(20);
        try {
            CountDownLatch ready = new CountDownLatch(20);
            CountDownLatch go = new CountDownLatch(1);
            List<Future<CompletableFuture<Result>>> calls = new ArrayList<>();
            for (int i = 0; i < 20; ++i) {
                calls.add(
                        pool.submit(
                                () -> {
                                    ready.countDown();
                                    go.await();
                                    return h.metrics.scrapeSources(Collections.emptyList());
                                }));
            }
            Assert.assertTrue(ready.await(10, TimeUnit.SECONDS));
            go.countDown();
            List<CompletableFuture<Result>> results = new ArrayList<>();
            for (Future<CompletableFuture<Result>> call : calls) {
                results.add(call.get(10, TimeUnit.SECONDS));
            }
            // one fetch per source for 20 callers, still in flight
            Assert.assertEquals(h.fetcher.calls.size(), 2);
            for (CompletableFuture<Result> result : results) {
                Assert.assertFalse(result.isDone());
            }

            h.fetcher.complete(path("gpu0"), W0);
            h.fetcher.complete(path("gpu1"), W1);
            for (CompletableFuture<Result> result : results) {
                Result value = result.get(10, TimeUnit.SECONDS);
                Assert.assertEquals(value.body, MERGED);
                Assert.assertEquals(value.registered, 2);
                Assert.assertEquals(value.ok, 2);
            }
            Assert.assertEquals(h.fetcher.calls.size(), 2);
        } finally {
            pool.shutdownNow();
        }
    }

    @Test
    public void testReuseForOneSecond() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 2);

        h.clock.addAndGet(SECOND - 1);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 2);

        h.clock.addAndGet(1);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 4);

        // a caller 1.1 s after the last fetch finished causes a new one
        h.clock.addAndGet(SECOND + SECOND / 10);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 6);
    }

    @Test
    public void testReuseWindowStartsWhenTheFetchFinishes() throws Exception {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.hold = true;
        CompletableFuture<Result> slow = h.metrics.scrapeSources(Collections.emptyList());
        Assert.assertEquals(h.fetcher.calls.size(), 2);

        // the fetch takes 1.5 s, longer than the reuse window
        h.clock.addAndGet(SECOND + SECOND / 2);
        h.fetcher.complete(path("gpu0"), W0);
        h.fetcher.complete(path("gpu1"), W1);
        Assert.assertEquals(slow.get(10, TimeUnit.SECONDS).body, MERGED);

        h.fetcher.hold = false;
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        h.clock.addAndGet(SECOND - 1);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 2);

        h.clock.addAndGet(1);
        Assert.assertEquals(h.scrape().body, MERGED);
        Assert.assertEquals(h.fetcher.calls.size(), 4);
    }

    @Test
    public void testReuseSharesRawBodiesNotResults() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        Result first = h.scrape();
        Result second = h.scrape();
        Assert.assertEquals(h.fetcher.calls.size(), 2);
        Assert.assertNotSame(first, second);
        Assert.assertEquals(first.body, second.body);
    }

    @Test
    public void testDifferentNamesDoNotShareAResult() {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.scrape(Collections.singletonList("vllm:running"));
        h.scrape(Collections.singletonList("vllm:running"));
        Assert.assertEquals(h.fetcher.calls.size(), 1);

        h.scrape(Collections.singletonList("vllm:requests_total"));
        Assert.assertEquals(h.fetcher.calls.size(), 2);

        // the order and repeats of the names do not matter
        h.scrape(Arrays.asList("b", "a", "b"));
        Assert.assertEquals(h.fetcher.calls.size(), 3);
        h.scrape(Arrays.asList("a", "b"));
        Assert.assertEquals(h.fetcher.calls.size(), 3);
        Assert.assertEquals(
                h.fetcher.calls.get(2), path("gpu0") + " /metrics?name%5B%5D=a&name%5B%5D=b");
    }

    @Test
    public void testNewRegistrationStartsANewFetch() {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.scrape();
        Assert.assertEquals(h.fetcher.calls.size(), 1);

        h.sources.put("gpu0", path("restarted"));
        h.fetcher.bodies.put(path("restarted"), W0);
        h.scrape();
        Assert.assertEquals(h.fetcher.calls.size(), 2);
        Assert.assertEquals(h.fetcher.calls.get(1), path("restarted") + " /metrics");
    }

    @Test
    public void testUri() {
        Assert.assertEquals(EngineMetrics.uri("gpu0", false, Collections.emptyList()), "/metrics");
        Assert.assertEquals(
                EngineMetrics.uri("gpu0", true, Collections.emptyList()), "/metrics?worker=gpu0");
        Assert.assertEquals(
                EngineMetrics.uri("gpu0-1", false, Collections.singletonList("vllm:running")),
                "/metrics?name%5B%5D=vllm%3Arunning");
        Assert.assertEquals(
                EngineMetrics.uri("gpu0", true, Arrays.asList("a", "b&c=d e", "ü")),
                "/metrics?worker=gpu0&name%5B%5D=a&name%5B%5D=b%26c%3Dd+e&name%5B%5D=%C3%BC");
        Assert.assertEquals(
                EngineMetrics.uri("a b/c", true, Collections.emptyList()),
                "/metrics?worker=a+b%2Fc");
    }

    @Test
    public void testNamesAreTheOnlyForwardedParameters() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        h.scrape(Arrays.asList("z", "vllm:a_b", "vllm:a b"));
        Assert.assertEquals(h.fetcher.calls.size(), 2);
        for (String call : h.fetcher.calls) {
            String uri = call.substring(call.indexOf(' ') + 1);
            Assert.assertTrue(
                    uri.matches("/metrics\\?worker=gpu[01]&name%5B%5D=vllm%3Aa_b&name%5B%5D=z"),
                    uri);
        }
    }

    @Test
    public void testFetchRefusesALinkThatLeavesTheTempDirectory() throws IOException {
        Path tmp = Paths.get(System.getProperty("java.io.tmpdir")).toRealPath();
        Path outsideDir = Paths.get("build", "engine-metrics-outside").toAbsolutePath();
        Files.createDirectories(outsideDir);
        if (outsideDir.toRealPath().startsWith(tmp)) {
            throw new SkipException("the build directory is inside the temporary directory");
        }
        Path target = outsideDir.resolve("djl_sock.0.metrics");
        Files.write(target, new byte[0]);
        Path link = tmp.resolve("djl_sock.link-" + System.nanoTime() + ".metrics");
        try {
            Files.createSymbolicLink(link, target);
        } catch (IOException | UnsupportedOperationException e) {
            throw new SkipException("symbolic links are not available", e);
        }
        try {
            CompletableFuture<String> fetched = EngineMetrics.fetch(link.toString(), "/metrics");
            ExecutionException e =
                    Assert.expectThrows(ExecutionException.class, () -> fetched.get());
            Assert.assertTrue(e.getCause() instanceof IOException, String.valueOf(e.getCause()));
            Assert.assertTrue(
                    e.getCause().getMessage().contains("outside the temporary directory"),
                    e.getCause().getMessage());
        } finally {
            Files.deleteIfExists(link);
            Files.deleteIfExists(target);
        }
    }

    @Test
    public void testOnlyInvalidNamesFetchNothing() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        Result result = h.scrape(Arrays.asList("vllm:a b", "x{y}"));
        Assert.assertTrue(h.fetcher.calls.isEmpty(), String.valueOf(h.fetcher.calls));
        Assert.assertEquals(result.body, "");
        Assert.assertEquals(result.registered, 0);
        Assert.assertEquals(result.ok, 0);
    }

    @Test
    public void testSourcesReadRegistrations() {
        Path tmp = tmpDir();
        Map<String, String> first = new HashMap<>();
        first.put("engine_metrics_socket.gpu0", socket(tmp, "djl_sock.18080.metrics"));
        first.put("engine_metrics_socket.gpu0-1", socket(tmp, "djl_sock.18081.metrics"));
        first.put("engine_metrics_socket.cpu", socket(tmp, "djl_sock.18083.metrics"));
        first.put("unrelated", "x");
        Map<String, String> second = new HashMap<>();
        second.put("engine_metrics_socket.gpu1", socket(tmp, "djl_sock.18082.metrics"));
        second.put("engine_metrics_socket.nc0", socket(tmp, "djl_sock.18084.metrics"));
        second.put(
                "engine_metrics_socket." + "x".repeat(64), socket(tmp, "djl_sock.18085.metrics"));
        Endpoint one = endpoint("m", "1", model(first));
        Endpoint two = endpoint("n", "1", model(second));

        SortedMap<String, String> expected = new TreeMap<>();
        expected.put("gpu0", socket(tmp, "djl_sock.18080.metrics"));
        expected.put("gpu0-1", socket(tmp, "djl_sock.18081.metrics"));
        expected.put("gpu1", socket(tmp, "djl_sock.18082.metrics"));
        expected.put("cpu", socket(tmp, "djl_sock.18083.metrics"));
        expected.put("nc0", socket(tmp, "djl_sock.18084.metrics"));
        expected.put("x".repeat(64), socket(tmp, "djl_sock.18085.metrics"));
        SortedMap<String, String> found = EngineMetrics.sources(Arrays.asList(one, two));
        Assert.assertEquals(found, expected);
        // sorted by label
        Assert.assertEquals(
                new ArrayList<>(found.keySet()),
                Arrays.asList("cpu", "gpu0", "gpu0-1", "gpu1", "nc0", "x".repeat(64)));

        // an endpoint listed twice gives each source once
        Assert.assertEquals(EngineMetrics.sources(Arrays.asList(one, one)).size(), 3);
        Assert.assertTrue(EngineMetrics.sources(Collections.emptyList()).isEmpty());
    }

    @Test
    public void testSourcesIgnoreUntrustedPaths() {
        Path tmp = tmpDir();
        Map<String, String> bad = new HashMap<>();
        bad.put("engine_metrics_socket.outside", "/etc/passwd.metrics");
        bad.put("engine_metrics_socket.suffix", tmp + "/djl_sock.1");
        bad.put("engine_metrics_socket.traversal", tmp + "/../etc/x.metrics");
        bad.put("engine_metrics_socket.relative", "x.metrics");
        bad.put("engine_metrics_socket.sibling", tmp + "x/y.metrics");
        bad.put("engine_metrics_socket.", tmp + "/empty.metrics");
        bad.put("engine_metrics_socket.new\nline", tmp + "/a.metrics");
        bad.put("engine_metrics_socket.sp ace", tmp + "/b.metrics");
        bad.put("engine_metrics_socket.sl/ash", tmp + "/c.metrics");
        bad.put("engine_metrics_socket.br{ace}", tmp + "/d.metrics");
        bad.put("engine_metrics_socket." + "x".repeat(65), tmp + "/e.metrics");
        bad.put("engine_metrics_socket.dir", tmp + "/");
        bad.put("engine_metrics_socket.nul", tmp + "/a\u0000b.metrics");
        bad.put("engine_metrics_socket.good", tmp + "/dir/./good.metrics");
        Endpoint endpoint = endpoint("m", "1", model(bad));
        SortedMap<String, String> found =
                EngineMetrics.sources(Collections.singletonList(endpoint));
        // the path that is checked is the path that is used
        Assert.assertEquals(
                found,
                Collections.singletonMap(
                        "good", tmp.resolve("dir").resolve("good.metrics").toString()));
    }

    @Test
    public void testRelativePathsAreIgnoredWhateverTheWorkingDirectoryIs() {
        String name = "java.io.tmpdir";
        String old = System.getProperty(name);
        try {
            // A build in a directory under the temporary directory has the working directory
            // inside it. A relative path must not be resolved against that directory.
            Path cwd = Path.of("").toAbsolutePath().normalize();
            System.setProperty(name, cwd.toString());
            Map<String, String> properties = new HashMap<>();
            properties.put("engine_metrics_socket.plain", "x.metrics");
            properties.put("engine_metrics_socket.dotted", "./y.metrics");
            properties.put("engine_metrics_socket.nested", "dir/z.metrics");
            properties.put("engine_metrics_socket.absolute", cwd.resolve("w.metrics").toString());
            Endpoint endpoint = endpoint("m", "1", model(properties));
            Assert.assertEquals(
                    EngineMetrics.sources(Collections.singletonList(endpoint)),
                    Collections.singletonMap("absolute", cwd.resolve("w.metrics").toString()));
        } finally {
            restore(name, old);
        }
    }

    @Test
    public void testSourcesSkipModelsWithoutProperties() {
        Endpoint endpoint = new Endpoint();
        ModelInfo<Input, Output> info = new ModelInfo<>("m");
        Assert.assertTrue(endpoint.add(new Workflow(info)));
        Assert.assertTrue(EngineMetrics.sources(Collections.singletonList(endpoint)).isEmpty());
        info.getModels().put(Device.cpu(), model(null));
        Assert.assertTrue(EngineMetrics.sources(Collections.singletonList(endpoint)).isEmpty());
    }

    @Test
    public void testSourcesTakeTheEndpointLock() throws Exception {
        Path tmp = tmpDir();
        Map<String, String> properties = new HashMap<>();
        properties.put("engine_metrics_socket.gpu0", tmp + "/djl_sock.1.metrics");
        Endpoint endpoint = endpoint("m", "1", model(properties));
        CountDownLatch locked = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<?> holder =
                    pool.submit(
                            () -> {
                                synchronized (endpoint) {
                                    locked.countDown();
                                    release.await();
                                }
                                return null;
                            });
            Assert.assertTrue(locked.await(10, TimeUnit.SECONDS));
            Future<SortedMap<String, String>> reader =
                    pool.submit(() -> EngineMetrics.sources(Collections.singletonList(endpoint)));
            try {
                reader.get(300, TimeUnit.MILLISECONDS);
                Assert.fail("sources() did not wait for the endpoint lock");
            } catch (TimeoutException expected) {
                // blocked on the lock, as intended
            }
            release.countDown();
            Assert.assertEquals(reader.get(10, TimeUnit.SECONDS).size(), 1);
            holder.get(10, TimeUnit.SECONDS);
        } finally {
            release.countDown();
            pool.shutdownNow();
        }
    }

    @Test
    public void testSourcesWhileWorkflowsAreAddedAndRemoved() throws Exception {
        Path tmp = tmpDir();
        Map<String, String> properties = new HashMap<>();
        properties.put("engine_metrics_socket.gpu0", tmp + "/djl_sock.1.metrics");
        ZooModel<Input, Output> model = model(properties);
        Endpoint endpoint = new Endpoint();
        AtomicBoolean done = new AtomicBoolean();
        ExecutorService pool = Executors.newFixedThreadPool(2);
        try {
            Future<Integer> writer =
                    pool.submit(
                            () -> {
                                int changes = 0;
                                for (int i = 0; i < 3000; ++i) {
                                    String version = "v" + (i % 5);
                                    ModelInfo<Input, Output> info = info("m", version);
                                    info.getModels().put(Device.cpu(), model);
                                    if (endpoint.add(new Workflow(info))) {
                                        ++changes;
                                    }
                                    if (i % 3 == 0
                                            && endpoint.remove("v" + ((i + 2) % 5)) != null) {
                                        ++changes;
                                    }
                                }
                                done.set(true);
                                return changes;
                            });
            Future<Integer> reader =
                    pool.submit(
                            () -> {
                                int reads = 0;
                                while (!done.get() || reads < 100) {
                                    SortedMap<String, String> found =
                                            EngineMetrics.sources(
                                                    Collections.singletonList(endpoint));
                                    Assert.assertTrue(found.size() <= 1);
                                    ++reads;
                                }
                                return reads;
                            });
            Assert.assertTrue(writer.get(60, TimeUnit.SECONDS) > 0);
            Assert.assertTrue(reader.get(60, TimeUnit.SECONDS) >= 100);
        } finally {
            done.set(true);
            pool.shutdownNow();
        }
    }

    @Test
    public void testLoggingIsRateLimited() {
        Harness h = new Harness(sources("gpu0", "gpu1"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.bodies.put(path("gpu1"), W1);
        h.scrape();
        Assert.assertTrue(h.logs.isEmpty(), h.logs.toString());

        // first failure: one warning with the reason
        h.fetcher.failing.add(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 1, h.logs.toString());
        Assert.assertEquals(h.logs.get(0), "WARN Engine metrics source gpu1 failed: timeout");

        // repeats stay quiet for five minutes
        for (int i = 0; i < 29; ++i) {
            h.clock.addAndGet(10 * SECOND);
            h.scrape();
        }
        Assert.assertEquals(h.logs.size(), 1, h.logs.toString());

        // then one reminder with the number of failed fetches since the last message
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());
        Assert.assertEquals(
                h.logs.get(1),
                "WARN Engine metrics source gpu1 is still failing: timeout (30 failed fetches"
                        + " since the last message)");
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());

        // recovery is logged once at INFO
        h.fetcher.failing.remove(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 3, h.logs.toString());
        Assert.assertEquals(h.logs.get(2), "INFO Engine metrics source gpu1 recovered");
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 3, h.logs.toString());

        // a failure shortly after the last warning is not reported again, and nor is its recovery
        h.fetcher.failing.add(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 3, h.logs.toString());
        h.fetcher.failing.remove(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 3, h.logs.toString());

        // a failure after the interval is reported again, with the failures that were not
        h.fetcher.failing.add(path("gpu1"));
        h.clock.addAndGet(EngineMetrics.LOG_INTERVAL_NANOS);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 4, h.logs.toString());
        Assert.assertEquals(
                h.logs.get(3),
                "WARN Engine metrics source gpu1 failed: timeout (3 failed fetches since the"
                        + " last message)");
        h.fetcher.failing.remove(path("gpu1"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 5, h.logs.toString());
        Assert.assertEquals(h.logs.get(4), "INFO Engine metrics source gpu1 recovered");
    }

    @Test
    public void testFlappingSourceIsReportedOncePerInterval() {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.scrape();
        Assert.assertTrue(h.logs.isEmpty(), h.logs.toString());

        // ten cycles of a failed and a good scrape, one scrape every ten seconds
        for (int i = 0; i < 10; ++i) {
            h.fetcher.failing.add(path("gpu0"));
            h.clock.addAndGet(10 * SECOND);
            h.scrape();
            h.fetcher.failing.remove(path("gpu0"));
            h.clock.addAndGet(10 * SECOND);
            h.scrape();
        }
        Assert.assertEquals(
                h.logs,
                Arrays.asList(
                        "WARN Engine metrics source gpu0 failed: timeout",
                        "INFO Engine metrics source gpu0 recovered"));

        // a source that stays down is reminded of after the interval
        h.fetcher.failing.add(path("gpu0"));
        for (int i = 0; i < 31; ++i) {
            h.clock.addAndGet(10 * SECOND);
            h.scrape();
        }
        Assert.assertEquals(h.logs.size(), 3, h.logs.toString());
        Assert.assertTrue(
                h.logs.get(2).startsWith("WARN Engine metrics source gpu0 is still failing"),
                h.logs.get(2));
    }

    @Test
    public void testRecoveryIsLoggedOnlyForAFailureThatWasLogged() {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.scrape();
        h.fetcher.failing.add(path("gpu0"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        h.fetcher.failing.remove(path("gpu0"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());

        // this failure is inside the interval of the warning above: it is not logged and the
        // source is not reported as recovered from something that was never reported
        h.fetcher.failing.add(path("gpu0"));
        for (int i = 0; i < 5; ++i) {
            h.clock.addAndGet(10 * SECOND);
            h.scrape();
        }
        h.fetcher.failing.remove(path("gpu0"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());
    }

    @Test
    public void testLoggingConcurrentScrapesCountOnce() throws Exception {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.hold = true;
        List<CompletableFuture<Result>> results = new ArrayList<>();
        for (int i = 0; i < 10; ++i) {
            results.add(h.metrics.scrapeSources(Collections.emptyList()));
        }
        h.fetcher.completeExceptionally(path("gpu0"), new TimeoutException("slow"));
        for (CompletableFuture<Result> result : results) {
            Assert.assertEquals(result.get(10, TimeUnit.SECONDS).ok, 0);
        }
        Assert.assertEquals(
                h.logs,
                Collections.singletonList("WARN Engine metrics source gpu0 failed: timeout"));
    }

    @Test
    public void testLoggingForgetsSourcesThatAreGone() {
        Harness h = new Harness(sources("gpu0"), false);
        h.fetcher.failing.add(path("gpu0"));
        h.scrape();
        Assert.assertEquals(h.logs.size(), 1);

        h.sources.clear();
        h.clock.addAndGet(10 * SECOND);
        h.scrape();

        // the same label registers again and fails: a first failure again
        h.sources.put("gpu0", path("gpu0"));
        h.clock.addAndGet(10 * SECOND);
        h.scrape();
        Assert.assertEquals(h.logs.size(), 2, h.logs.toString());
    }

    @Test
    public void testLoggingReasonsPerSource() {
        Harness h = new Harness(sources("gpu0", "gpu1", "gpu2"), false);
        h.fetcher.bodies.put(path("gpu0"), W0);
        h.fetcher.errors.put(
                path("gpu1"),
                new EngineMetricsClient.HttpStatusException(HttpResponseStatus.NOT_FOUND));
        h.fetcher.errors.put(path("gpu2"), new NoSuchFileException("/tmp/x"));
        h.scrape();
        Assert.assertEquals(
                h.logs,
                Arrays.asList(
                        "WARN Engine metrics source gpu1 failed: HTTP status 404",
                        "WARN Engine metrics source gpu2 failed: socket file missing"));
    }

    @Test
    public void testOversizeAndTimeoutAgainstSockets() throws Exception {
        if (!UnixSocketTestServer.isSupported()) {
            throw new SkipException("Unix domain sockets need Epoll or KQueue");
        }
        String big = "x".repeat(EngineMetrics.MAX_BODY_BYTES + 1);
        try (UnixSocketTestServer good = UnixSocketTestServer.start(UnixSocketTestServer.ok(W0));
                UnixSocketTestServer large =
                        UnixSocketTestServer.start(UnixSocketTestServer.ok(big));
                UnixSocketTestServer hung =
                        UnixSocketTestServer.start(UnixSocketTestServer.silent());
                UnixSocketTestServer closing =
                        UnixSocketTestServer.start(UnixSocketTestServer.closing())) {
            SortedMap<String, String> registered = new TreeMap<>();
            registered.put("gpu0", good.path());
            registered.put("gpu1", large.path());
            registered.put("gpu2", hung.path());
            registered.put("gpu3", closing.path());
            registered.put("gpu4", good.path() + ".gone");
            List<String> logs = new CopyOnWriteArrayList<>();
            EngineMetrics metrics =
                    new EngineMetrics(
                            () -> registered,
                            () -> false,
                            EngineMetrics::fetch,
                            System::nanoTime,
                            recordingLogger(logs));
            long begin = System.nanoTime();
            Result result =
                    metrics.scrapeSources(Collections.emptyList()).get(20, TimeUnit.SECONDS);
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);

            Assert.assertEquals(result.registered, 5);
            Assert.assertEquals(result.ok, 1);
            Assert.assertEquals(result.body, W0);
            // the hung worker is given up on after about two seconds, and not much later
            Assert.assertTrue(elapsed >= 1500 && elapsed < 6000, "elapsed " + elapsed);
            Assert.assertEquals(
                    new ArrayList<>(logs).stream().sorted().toArray(),
                    new Object[] {
                        "WARN Engine metrics source gpu1 failed: response over the size limit",
                        "WARN Engine metrics source gpu2 failed: timeout",
                        "WARN Engine metrics source gpu3 failed: closed before the response",
                        "WARN Engine metrics source gpu4 failed: socket file missing"
                    });
            // the worker that answered saw the label request
            Assert.assertEquals(
                    good.requests(),
                    Collections.singletonList("GET /metrics?worker=gpu0 | localhost | close"));
        }
    }

    @Test
    public void testSocketsAreFetchedConcurrently() throws Exception {
        if (!UnixSocketTestServer.isSupported()) {
            throw new SkipException("Unix domain sockets need Epoll or KQueue");
        }
        try (UnixSocketTestServer first =
                        UnixSocketTestServer.start(UnixSocketTestServer.delayed(1000, W0));
                UnixSocketTestServer second =
                        UnixSocketTestServer.start(UnixSocketTestServer.delayed(1000, W1))) {
            SortedMap<String, String> registered = new TreeMap<>();
            registered.put("gpu0", first.path());
            registered.put("gpu1", second.path());
            EngineMetrics metrics =
                    new EngineMetrics(
                            () -> registered,
                            () -> false,
                            EngineMetrics::fetch,
                            System::nanoTime,
                            mock(Logger.class));
            long begin = System.nanoTime();
            Result result =
                    metrics.scrapeSources(Collections.emptyList()).get(20, TimeUnit.SECONDS);
            long elapsed = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - begin);
            Assert.assertEquals(result.ok, 2);
            Assert.assertEquals(result.body, MERGED);
            // two answers of 1000 ms one after the other would take 2000 ms
            Assert.assertTrue(elapsed >= 950 && elapsed < 1900, "elapsed " + elapsed);
        }
    }

    @Test
    public void testMergeOfSocketAnswers() throws Exception {
        if (!UnixSocketTestServer.isSupported()) {
            throw new SkipException("Unix domain sockets need Epoll or KQueue");
        }
        try (UnixSocketTestServer first = UnixSocketTestServer.start(UnixSocketTestServer.ok(W0));
                UnixSocketTestServer second =
                        UnixSocketTestServer.start(UnixSocketTestServer.ok(W1))) {
            SortedMap<String, String> registered = new TreeMap<>();
            registered.put("gpu0", first.path());
            registered.put("gpu1", second.path());
            EngineMetrics metrics =
                    new EngineMetrics(
                            () -> registered,
                            () -> false,
                            EngineMetrics::fetch,
                            System::nanoTime,
                            mock(Logger.class));
            Result result =
                    metrics.scrapeSources(Collections.singletonList("vllm:running"))
                            .get(20, TimeUnit.SECONDS);
            Assert.assertEquals(result.body, MERGED);
            Assert.assertEquals(
                    first.requests(),
                    Collections.singletonList(
                            "GET /metrics?worker=gpu0&name%5B%5D=vllm%3Arunning | localhost |"
                                    + " close"));
            Assert.assertEquals(
                    second.requests(),
                    Collections.singletonList(
                            "GET /metrics?worker=gpu1&name%5B%5D=vllm%3Arunning | localhost |"
                                    + " close"));
        }
    }

    private static void restore(String name, String old) {
        if (old == null) {
            System.clearProperty(name);
        } else {
            System.setProperty(name, old);
        }
    }

    private static Path tmpDir() {
        return Path.of(System.getProperty("java.io.tmpdir")).toAbsolutePath().normalize();
    }

    private static String socket(Path dir, String name) {
        return dir.resolve(name).toString();
    }

    private static String path(String label) {
        return tmpDir() + "/djl_sock." + label + ".metrics";
    }

    private static SortedMap<String, String> sources(String... labels) {
        SortedMap<String, String> map = new TreeMap<>();
        for (String label : labels) {
            map.put(label, path(label));
        }
        return map;
    }

    @SuppressWarnings("unchecked")
    private static ZooModel<Input, Output> model(Map<String, String> properties) {
        ZooModel<Input, Output> model = mock(ZooModel.class);
        when(model.getProperties()).thenReturn(properties);
        return model;
    }

    private static ModelInfo<Input, Output> info(String id, String version) {
        return new ModelInfo<>(
                id,
                "file:///" + id,
                version,
                "Python",
                "",
                Input.class,
                Output.class,
                10,
                60,
                100,
                1,
                1,
                1);
    }

    private static Endpoint endpoint(String id, String version, ZooModel<Input, Output> model) {
        ModelInfo<Input, Output> info = info(id, version);
        info.getModels().put(Device.cpu(), model);
        Endpoint endpoint = new Endpoint();
        Assert.assertTrue(endpoint.add(new Workflow(info)));
        return endpoint;
    }

    private static Logger recordingLogger(List<String> logs) {
        return mock(
                Logger.class,
                invocation -> {
                    String level = invocation.getMethod().getName();
                    if ("warn".equals(level) || "info".equals(level)) {
                        Object[] arguments = invocation.getArguments();
                        Object[] parameters = Arrays.copyOfRange(arguments, 1, arguments.length);
                        String message =
                                MessageFormatter.arrayFormat((String) arguments[0], parameters)
                                        .getMessage();
                        logs.add(level.toUpperCase(Locale.ROOT) + " " + message);
                    }
                    return null;
                });
    }

    /** An engine metrics collector wired to fakes. */
    private static final class Harness {

        final SortedMap<String, String> sources;
        final AtomicLong clock = new AtomicLong(1_000_000_000_000L);
        final FakeFetcher fetcher = new FakeFetcher();
        final List<String> logs = new CopyOnWriteArrayList<>();
        final EngineMetrics metrics;

        Harness(SortedMap<String, String> registered, boolean multiModel) {
            sources = registered;
            metrics =
                    new EngineMetrics(
                            () -> new TreeMap<>(sources),
                            () -> multiModel,
                            fetcher,
                            clock::get,
                            recordingLogger(logs));
        }

        Result scrape() {
            return scrape(Collections.emptyList());
        }

        Result scrape(List<String> names) {
            try {
                return metrics.scrapeSources(names).get(10, TimeUnit.SECONDS);
            } catch (InterruptedException | ExecutionException | TimeoutException e) {
                throw new AssertionError(e);
            }
        }
    }

    /** Answers from memory, or holds the answers until the test completes them. */
    private static final class FakeFetcher implements EngineMetrics.Fetcher {

        final List<String> calls = new CopyOnWriteArrayList<>();
        final Map<String, String> bodies = new ConcurrentHashMap<>();
        final Map<String, Throwable> errors = new ConcurrentHashMap<>();
        final Set<String> failing = ConcurrentHashMap.newKeySet();
        final Set<String> throwing = ConcurrentHashMap.newKeySet();
        final Map<String, List<CompletableFuture<String>>> held = new ConcurrentHashMap<>();
        boolean hold;

        /** An error that the next call throws. */
        volatile Error fault;

        /** {@inheritDoc} */
        @Override
        public CompletableFuture<String> get(String socketPath, String uri) {
            calls.add(socketPath + " " + uri);
            Error thrown = fault;
            if (thrown != null) {
                fault = null;
                throw thrown;
            }
            if (throwing.contains(socketPath)) {
                throw new IllegalStateException("cannot start " + socketPath);
            }
            if (hold) {
                CompletableFuture<String> future = new CompletableFuture<>();
                held.computeIfAbsent(socketPath, k -> new CopyOnWriteArrayList<>()).add(future);
                return future;
            }
            if (failing.contains(socketPath)) {
                return CompletableFuture.failedFuture(new TimeoutException("no answer"));
            }
            Throwable error = errors.get(socketPath);
            if (error != null) {
                return CompletableFuture.failedFuture(error);
            }
            String body = bodies.get(socketPath);
            if (body == null) {
                return CompletableFuture.failedFuture(new IllegalStateException("none"));
            }
            return CompletableFuture.completedFuture(body);
        }

        void complete(String socketPath, String body) {
            for (CompletableFuture<String> future : held.get(socketPath)) {
                future.complete(body);
            }
        }

        void completeExceptionally(String socketPath, Throwable error) {
            for (CompletableFuture<String> future : held.get(socketPath)) {
                future.completeExceptionally(error);
            }
        }
    }
}
