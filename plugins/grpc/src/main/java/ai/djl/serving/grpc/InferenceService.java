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

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.grpc.proto.InferenceGrpc;
import ai.djl.serving.grpc.proto.InferenceRequest;
import ai.djl.serving.grpc.proto.InferenceResponse;
import ai.djl.serving.grpc.proto.PingResponse;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.JsonUtils;

import com.google.protobuf.ByteString;
import com.google.protobuf.Empty;

import io.grpc.Status;
import io.grpc.stub.ServerCallStreamObserver;
import io.grpc.stub.StreamObserver;
import io.netty.handler.codec.http.HttpResponseStatus;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.concurrent.TimeUnit;

class InferenceService extends InferenceGrpc.InferenceImplBase {

    private static final Logger logger = LoggerFactory.getLogger(InferenceService.class);

    private int chunkReadTime;

    InferenceService() {
        chunkReadTime = ConfigManager.getInstance().getChunkedReadTimeout();
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public void ping(Empty request, StreamObserver<PingResponse> responseObserver) {
        ((ServerCallStreamObserver<PingResponse>) responseObserver)
                .setOnCancelHandler(
                        () -> {
                            logger.warn("grpc client call already cancelled");
                            responseObserver.onError(
                                    Status.CANCELLED
                                            .withDescription("call already cancelled")
                                            .asRuntimeException());
                        });
        ModelManager.getInstance()
                .workerStatus()
                .thenAccept(
                        w -> {
                            boolean hasFailure = (boolean) w.get("hasFailure");
                            boolean hasPending = (boolean) w.get("hasPending");

                            int code;
                            if (hasFailure) {
                                logger.info(
                                        "PING FAILED: {}", JsonUtils.GSON.toJson(w.get("data")));
                                code = HttpResponseStatus.INTERNAL_SERVER_ERROR.code();
                            } else if (hasPending) {
                                if (ConfigManager.getInstance().allowsMultiStatus()) {
                                    code = HttpResponseStatus.MULTI_STATUS.code();
                                } else {
                                    code = HttpResponseStatus.OK.code();
                                }
                            } else {
                                code = HttpResponseStatus.OK.code();
                            }

                            PingResponse.Builder builder = PingResponse.newBuilder().setCode(code);
                            Map<String, StatusResponse> map =
                                    (Map<String, StatusResponse>) w.get("data");
                            for (Map.Entry<String, StatusResponse> entry : map.entrySet()) {
                                String value = entry.getValue().getStatus();
                                builder.putModelStatus(entry.getKey(), value);
                            }

                            responseObserver.onNext(builder.build());
                            responseObserver.onCompleted();
                        });
    }

    /** {@inheritDoc} */
    @Override
    public void predict(InferenceRequest request, StreamObserver<InferenceResponse> observer) {
        ServerCallStreamObserver<InferenceResponse> scsObserver =
                (ServerCallStreamObserver<InferenceResponse>) observer;
        scsObserver.setOnCancelHandler(
                () -> {
                    logger.debug("grpc client call already cancelled");
                    observer.onError(
                            io.grpc.Status.CANCELLED
                                    .withDescription("call already cancelled")
                                    .asRuntimeException());
                });
        String workflowName = request.getModelName();
        String version = request.getModelVersion();
        if (version.isEmpty()) {
            version = null;
        }
        ModelManager modelManager = ModelManager.getInstance();
        if (workflowName.isEmpty()) {
            workflowName = ModelManager.getInstance().getSingleStartupWorkflow().orElse("");
        }
        Workflow workflow = modelManager.getWorkflow(workflowName, version, true);
        if (workflow == null) {
            observer.onError(
                    new ModelNotFoundException("Model or workflow not found: " + workflowName));
            return;
        }

        Input input = parseInput(request);
        modelManager
                .runJob(workflow, input)
                .whenCompleteAsync(
                        (o, t) -> {
                            if (o != null) {
                                sendOutput(o, scsObserver);
                            }
                        })
                .exceptionally(
                        t -> {
                            observer.onError(t.getCause());
                            return null;
                        });
    }

    private Input parseInput(InferenceRequest req) {
        Input input = new Input();
        for (Map.Entry<String, ByteString> entry : req.getHeadersMap().entrySet()) {
            String key = entry.getKey();
            String value = entry.getValue().toString(StandardCharsets.UTF_8);
            input.addProperty(key, value);
        }
        input.add(req.getInput().toByteArray());
        return input;
    }

    private void sendOutput(Output output, ServerCallStreamObserver<InferenceResponse> observer) {
        if (observer.isCancelled()) {
            return;
        }

        BytesSupplier data = output.getData();
        if (data instanceof ChunkedBytesSupplier) {
            try {
                boolean first = true;
                ChunkedBytesSupplier supplier = (ChunkedBytesSupplier) data;
                while (supplier.hasNext()) {
                    InferenceResponse.Builder builder = InferenceResponse.newBuilder();
                    builder.setCode(output.getCode());
                    byte[] buf = supplier.nextChunk(chunkReadTime, TimeUnit.SECONDS);
                    builder.setOutput(ByteString.copyFrom(buf));
                    if (first) {
                        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
                            ByteString value =
                                    ByteString.copyFrom(entry.getValue(), StandardCharsets.UTF_8);
                            builder.putHeaders(entry.getKey(), value);
                        }
                        first = false;
                    }
                    observer.onNext(builder.build());
                }
            } catch (InterruptedException | IllegalStateException e) {
                logger.warn("Chunk reading interrupted", e);
            } finally {
                observer.onCompleted();
            }
            return;
        }

        InferenceResponse.Builder builder = InferenceResponse.newBuilder();
        builder.setCode(output.getCode());
        builder.setOutput(ByteString.copyFrom(output.getData().getAsBytes()));
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            ByteString value = ByteString.copyFrom(entry.getValue(), StandardCharsets.UTF_8);
            builder.putHeaders(entry.getKey(), value);
        }
        observer.onNext(builder.build());
        observer.onCompleted();
    }
}
