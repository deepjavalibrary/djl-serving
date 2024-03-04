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
import java.util.Collections;
import java.util.HashSet;
import java.util.List;

final class PrometheusExporter {

    private PrometheusExporter() {}

    static void handle(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        ByteBuf buf = Unpooled.directBuffer();
        List<String> params = decoder.parameters().getOrDefault("name[]", Collections.emptyList());
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
}
