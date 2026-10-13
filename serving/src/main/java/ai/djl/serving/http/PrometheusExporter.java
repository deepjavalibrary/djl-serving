/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.prometheus.MetricExporter;
import ai.djl.serving.util.NettyUtils;
import ai.djl.util.Utils;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.ByteBufOutputStream;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

final class PrometheusExporter {

    private PrometheusExporter() {}

    static void handle(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        List<String> params = decoder.parameters().getOrDefault("name[]", Collections.emptyList());
        if (EngineMetrics.isEnabled()) {
            handleWithEngineMetrics(ctx, params);
            return;
        }
        ByteBuf buf = Unpooled.directBuffer();
        try (OutputStream os = new ByteBufOutputStream(buf)) {
            MetricExporter.export(os, new HashSet<>(params));
        } catch (IllegalArgumentException e) {
            throw new BadRequestException(e.getMessage(), e);
        } catch (IOException e) {
            throw new InternalServerException("Failed to encode prometheus metrics", e);
        }
        FullHttpResponse resp =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, buf);
        resp.headers().set(HttpHeaderNames.CONTENT_TYPE, MetricExporter.CONTENT_TYPE);
        NettyUtils.sendHttpResponse(ctx, resp, true);
    }

    /**
     * Answers with the metrics of DJL Serving, if they are enabled, followed by the engine metrics
     * of the Python workers.
     *
     * <p>The answer is sent when the workers have answered. The request is released when the caller
     * returns, so everything it holds is read here.
     */
    private static void handleWithEngineMetrics(ChannelHandlerContext ctx, List<String> names) {
        boolean djlMetrics =
                Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_PROMETHEUS"));
        respond(ctx, djlMetrics, new HashSet<>(names), EngineMetrics.scrape(names));
    }

    /**
     * Sends the answer when the scrape is done, on the event loop of the channel.
     *
     * <p>The thread that completes a scrape is the one thread of the metrics client, which also
     * runs the connects, reads and deadlines of every fetch. Building the response and writing it
     * (an access log line, for example) there would hold up the fetches of the other workers.
     */
    static void respond(
            ChannelHandlerContext ctx,
            boolean djlMetrics,
            Set<String> filter,
            CompletableFuture<EngineMetrics.Result> scrape) {
        scrape.whenCompleteAsync(
                (result, error) -> sendEngineMetrics(ctx, djlMetrics, filter, result, error),
                ctx.executor());
    }

    private static void sendEngineMetrics(
            ChannelHandlerContext ctx,
            boolean djlMetrics,
            Set<String> filter,
            EngineMetrics.Result result,
            Throwable error) {
        try {
            if (error != null) {
                boolean wrapped = error instanceof CompletionException && error.getCause() != null;
                Throwable cause = wrapped ? error.getCause() : error;
                if (cause instanceof ServiceUnavailableException) {
                    // refused on purpose, and EngineMetrics logs that once in a while: a stack
                    // trace for each scrape of a flood would only add to it
                    NettyUtils.sendError(ctx, HttpResponseStatus.SERVICE_UNAVAILABLE, cause);
                } else {
                    NettyUtils.sendError(ctx, cause);
                }
                return;
            }
            if (result.registered > 0 && result.ok == 0) {
                // a scraper has to see a failed scrape, like it does when the engine is down
                String msg =
                        "Engine metrics are unavailable: none of the "
                                + result.registered
                                + " registered workers answered.";
                NettyUtils.sendError(
                        ctx,
                        HttpResponseStatus.SERVICE_UNAVAILABLE,
                        new ServiceUnavailableException(msg));
                return;
            }
            ByteBuf buf = Unpooled.buffer(result.body.length() + 1024);
            if (djlMetrics) {
                try (OutputStream os = new ByteBufOutputStream(buf)) {
                    MetricExporter.export(os, filter);
                }
            }
            buf.writeCharSequence(result.body, StandardCharsets.UTF_8);
            FullHttpResponse resp =
                    new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, buf);
            resp.headers().set(HttpHeaderNames.CONTENT_TYPE, MetricExporter.CONTENT_TYPE);
            NettyUtils.sendHttpResponse(ctx, resp, true);
        } catch (IOException | RuntimeException e) {
            // nobody looks at the future that this runs in
            NettyUtils.sendError(ctx, e);
        }
    }
}
