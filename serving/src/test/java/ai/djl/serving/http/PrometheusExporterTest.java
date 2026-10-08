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

import ai.djl.serving.Arguments;
import ai.djl.serving.http.EngineMetrics.Result;
import ai.djl.serving.util.ConfigManager;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.util.CharsetUtil;

import org.apache.commons.cli.CommandLine;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.Collections;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

public class PrometheusExporterTest {

    private static final String BODY = "# HELP vllm:a A.\n# TYPE vllm:a gauge\nvllm:a 1.0\n";

    @BeforeClass
    public void setUp() {
        if (ConfigManager.getInstance() == null) {
            ConfigManager.init(new Arguments(CommandLine.builder().build()));
        }
    }

    @Test
    public void testAnswerIsSentOnTheEventLoopOfTheChannel() throws InterruptedException {
        EmbeddedChannel channel = new EmbeddedChannel(new ChannelInboundHandlerAdapter());
        CompletableFuture<Result> scrape = new CompletableFuture<>();
        PrometheusExporter.respond(context(channel), false, Collections.emptySet(), scrape);

        // The metrics client completes a scrape on its one thread, which also runs the deadlines
        // of every fetch. That thread must not build the answer or write it.
        Thread client =
                new Thread(
                        () -> scrape.complete(new Result(BODY, 1, 1)), "djl-engine-metrics-test");
        client.start();
        client.join();
        Assert.assertNull(channel.readOutbound(), "the client thread wrote the answer");

        channel.runPendingTasks();
        FullHttpResponse response = channel.readOutbound();
        Assert.assertNotNull(response, "the answer was not sent on the event loop");
        Assert.assertEquals(response.status(), HttpResponseStatus.OK);
        Assert.assertEquals(response.content().toString(CharsetUtil.UTF_8), BODY);
        Assert.assertTrue(
                response.headers().get(HttpHeaderNames.CONTENT_TYPE).startsWith("text/plain"));
        response.release();
        channel.finishAndReleaseAll();
    }

    @Test
    public void testAnswerOfAFinishedScrapeIsSentOnTheEventLoopToo() {
        EmbeddedChannel channel = new EmbeddedChannel(new ChannelInboundHandlerAdapter());
        CompletableFuture<Result> scrape =
                CompletableFuture.completedFuture(new Result(BODY, 1, 1));
        PrometheusExporter.respond(context(channel), false, Collections.emptySet(), scrape);

        channel.runPendingTasks();
        FullHttpResponse response = channel.readOutbound();
        Assert.assertNotNull(response);
        Assert.assertEquals(response.status(), HttpResponseStatus.OK);
        response.release();
        channel.finishAndReleaseAll();
    }

    @Test
    public void testNoRegisteredWorkerGivesAnEmptyAnswer() {
        FullHttpResponse response = answer(CompletableFuture.completedFuture(new Result("", 0, 0)));
        Assert.assertEquals(response.status(), HttpResponseStatus.OK);
        Assert.assertEquals(response.content().toString(CharsetUtil.UTF_8), "");
        response.release();
    }

    @Test
    public void testNoAnsweringWorkerIsAFailedScrape() {
        FullHttpResponse response = answer(CompletableFuture.completedFuture(new Result("", 2, 0)));
        Assert.assertEquals(response.status(), HttpResponseStatus.SERVICE_UNAVAILABLE);
        response.release();
    }

    @Test
    public void testFailedScrapeIsAnError() {
        // the error that a dependent stage reports is unwrapped: an illegal argument is a conflict
        CompletableFuture<Result> scrape = new CompletableFuture<>();
        scrape.completeExceptionally(new CompletionException(new IllegalArgumentException("bad")));
        FullHttpResponse response = answer(scrape);
        Assert.assertEquals(response.status(), HttpResponseStatus.CONFLICT);
        response.release();

        CompletableFuture<Result> failed = new CompletableFuture<>();
        failed.completeExceptionally(new OutOfMemoryError("unable to create native thread"));
        response = answer(failed);
        Assert.assertEquals(response.status(), HttpResponseStatus.INTERNAL_SERVER_ERROR);
        response.release();
    }

    @Test
    public void testRefusedScrapeIsServiceUnavailable() {
        String message = "Engine metrics are busy: 32 fetches are unfinished, try again later.";
        CompletableFuture<Result> refused =
                CompletableFuture.failedFuture(new ServiceUnavailableException(message));
        FullHttpResponse response = answer(refused);
        Assert.assertEquals(response.status(), HttpResponseStatus.SERVICE_UNAVAILABLE);
        Assert.assertTrue(response.content().toString(CharsetUtil.UTF_8).contains(message));
        response.release();

        // the same when a dependent stage reports it
        CompletableFuture<Result> wrapped = new CompletableFuture<>();
        wrapped.completeExceptionally(
                new CompletionException(new ServiceUnavailableException(message)));
        response = answer(wrapped);
        Assert.assertEquals(response.status(), HttpResponseStatus.SERVICE_UNAVAILABLE);
        response.release();
    }

    private static FullHttpResponse answer(CompletableFuture<Result> scrape) {
        EmbeddedChannel channel = new EmbeddedChannel(new ChannelInboundHandlerAdapter());
        PrometheusExporter.respond(context(channel), false, Collections.emptySet(), scrape);
        channel.runPendingTasks();
        FullHttpResponse response = channel.readOutbound();
        Assert.assertNotNull(response);
        channel.finishAndReleaseAll();
        return response;
    }

    private static ChannelHandlerContext context(EmbeddedChannel channel) {
        return channel.pipeline().firstContext();
    }
}
