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
package ai.djl.serving.grpc;

import ai.djl.serving.grpc.proto.InferenceGrpc;
import ai.djl.serving.grpc.proto.InferenceRequest;
import ai.djl.serving.grpc.proto.InferenceResponse;
import ai.djl.serving.grpc.proto.PingResponse;

import com.google.protobuf.ByteString;
import com.google.protobuf.Empty;

import io.grpc.Grpc;
import io.grpc.InsecureChannelCredentials;
import io.grpc.ManagedChannel;

import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.Iterator;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/** The gRPC client that connect to server. */
public class GrpcClient implements AutoCloseable {

    private ManagedChannel channel;
    private InferenceGrpc.InferenceBlockingStub stub;

    /**
     * Constructs client for accessing HelloWorld server using the existing channel.
     *
     * @param channel the managed channel
     */
    public GrpcClient(ManagedChannel channel) {
        this.channel = channel;
        stub = InferenceGrpc.newBlockingStub(channel);
    }

    /**
     * Constructs a new instance with target address.
     *
     * @param target the target address
     * @return a new instance
     */
    public static GrpcClient newInstance(String target) {
        ManagedChannel channel =
                Grpc.newChannelBuilder(target, InsecureChannelCredentials.create()).build();
        return new GrpcClient(channel);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        try {
            channel.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException ignore) {
            // ignore
        }
    }

    /**
     * Sends the {@code Ping} command to the server.
     *
     * @return the ping response
     */
    public PingResponse ping() {
        Empty request = Empty.getDefaultInstance();
        return stub.ping(request);
    }

    /**
     * Sends the {@code Ping} command to the server.
     *
     * @param modelName the model name
     * @param data the inference payload
     * @return the inference responses
     */
    public Iterator<InferenceResponse> inference(String modelName, String data) {
        return inference(modelName, null, Collections.emptyMap(), data);
    }

    /**
     * Sends the {@code Ping} command to the server.
     *
     * @param modelName the model name
     * @param version the model version
     * @param data the inference payload
     * @param headers the input headers
     * @return the inference responses
     */
    public Iterator<InferenceResponse> inference(
            String modelName, String version, Map<String, String> headers, String data) {
        InferenceRequest.Builder builder = InferenceRequest.newBuilder();
        if (modelName != null) {
            builder.setModelName(modelName);
        }
        if (version != null) {
            builder.setModelVersion(version);
        }
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            ByteString value = ByteString.copyFrom(entry.getValue(), StandardCharsets.UTF_8);
            builder.putHeaders(entry.getKey(), value);
        }

        ByteString input = ByteString.copyFrom(data, StandardCharsets.UTF_8);
        InferenceRequest req = builder.setInput(input).build();
        return stub.predict(req);
    }
}
