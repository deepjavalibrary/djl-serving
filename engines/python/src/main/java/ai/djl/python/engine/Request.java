/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.python.engine;

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metrics;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.JsonUtils;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

class Request {

    Input input;
    ChunkedBytesSupplier data;
    Output output;
    String nextToken;
    boolean last;
    String seed;
    Metrics metrics;
    Dimension dimension;
    int count;
    long creationTime;
    String requestId;

    Request(Input input, String seed, Metrics metrics, Dimension dimension) {
        this.input = input;
        data = new ChunkedBytesSupplier();
        output = new Output();
        output.add(data);
        this.seed = seed;
        this.metrics = metrics;
        this.dimension = dimension;
        creationTime = System.nanoTime();
        requestId = input.getProperty("requestId", "UNKNOWN_REQUEST_ID");
    }

    BytesSupplier getRequest() {
        if (nextToken != null) {
            return BytesSupplier.wrap("");
        }
        return input.getData();
    }

    Set<Map.Entry<String, String>> getProperties() {
        if (nextToken != null) {
            return Collections.emptySet();
        }
        return input.getProperties().entrySet();
    }

    /**
     * NextTokenChooserParameters is constructed during first forward and preserved for all forward
     * calls of the request.
     *
     * @return seed, only for first forward
     */
    String getSeed() {
        if (nextToken != null) {
            return null;
        }
        return seed;
    }

    String getRequestId() {
        return requestId;
    }

    void addResponse(byte[] json, Map<String, String> properties) {
        if (properties != null) {
            output.getProperties().putAll(properties);
        }
        ++count;
        ByteBuf buf = Unpooled.wrappedBuffer(json);
        int size = buf.readShort();
        String code = null;
        String error = null;
        for (int i = 0; i < size; ++i) {
            String key = Objects.requireNonNull(CodecUtils.readUtf8(buf));
            String value = Objects.requireNonNull(CodecUtils.readUtf8(buf));
            switch (key) {
                case "data":
                    nextToken = value;
                    break;
                case "last":
                    last = "true".equalsIgnoreCase(value);
                    break;
                case "code":
                    code = value;
                    break;
                case "error":
                    error = value;
                    break;
                default:
                    break;
            }
        }
        if ((nextToken == null || nextToken.isEmpty()) && !last) {
            // in non-streaming cases, we do not return content until generation is finished
            return;
        }
        if (code != null) {
            Map<String, Object> map = new ConcurrentHashMap<>(2);
            int httpStatusCode = Integer.parseInt(code);
            map.put("code", httpStatusCode);
            output.setCode(httpStatusCode);
            if (error != null) {
                map.put("error", error);
                output.setMessage(error);
            }
            byte[] buffer = JsonUtils.GSON.toJson(map).getBytes(StandardCharsets.UTF_8);
            data.appendContent(buffer, true);
        } else {
            if (last && metrics != null) {
                long duration = System.nanoTime() - creationTime;
                double throughput = count * 1_000_000_000d / duration;
                long latency = duration / count / 1000;
                metrics.addMetric("TokenLatency", latency, Unit.MICROSECONDS, dimension);
                metrics.addMetric("TokenThroughput", throughput, Unit.COUNT_PER_SECOND, dimension);
                metrics.addMetric("OutputTokens", count, Unit.COUNT_PER_ITEM, dimension);
            }
            data.appendContent(nextToken.getBytes(StandardCharsets.UTF_8), last);
        }
    }
}
