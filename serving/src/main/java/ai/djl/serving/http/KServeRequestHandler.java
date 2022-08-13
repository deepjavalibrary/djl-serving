/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Pair;

import com.google.gson.annotations.SerializedName;

import io.netty.buffer.ByteBufInputStream;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(KServeRequestHandler.class);

    private static final Pattern PATTERN = Pattern.compile("^/v2/.+");
    private static final String EMPTY_BODY = "";

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        HttpMethod method = req.method();
        try {
            if (isKServeDescribeModelReq(segments)) {
                requireGet(method);
                handleKServeDescribeModel(ctx, segments);
            } else if (isKServeDescribeHealthReadyReq(segments)
                    || isKServeDescribeHealthLiveReq(segments)) {
                requireGet(method);
                handleKServeDescribeHealth(ctx);
            } else if (isKServeDescribeModelReadyReq(segments)) {
                requireGet(method);
                handleKServeDescribeModelReady(ctx, segments);
            } else if (isKServeDescribeInferenceReq(segments)) {
                requirePost(method);
                inference(ctx, req, segments);
            } else {
                throw new ResourceNotFoundException();
            }
        } catch (Exception exception) {
            onException(exception, ctx);
        }
    }

    private void requireGet(HttpMethod method) {
        if (!HttpMethod.GET.equals(method)) {
            throw new MethodNotAllowedException();
        }
    }

    private void requirePost(HttpMethod method) {
        if (!HttpMethod.POST.equals(method)) {
            throw new MethodNotAllowedException();
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments) {
        return (segments.length == 4 || (segments.length == 6 && "version".equals(segments[4])))
                && "models".equals(segments[2]);
    }

    private boolean isKServeDescribeHealthLiveReq(String[] segments) {
        return segments.length == 4 && "health".equals(segments[2]) && "live".equals((segments[3]));
    }

    private boolean isKServeDescribeHealthReadyReq(String[] segments) {
        return segments.length == 4
                && "health".equals(segments[2])
                && "ready".equals((segments[3]));
    }

    private boolean isKServeDescribeModelReadyReq(String[] segments) {
        return (segments.length == 5 || (segments.length == 7 && "version".equals(segments[4])))
                && "models".equals(segments[2])
                && "ready".equals(segments[segments.length - 1]);
    }

    private boolean isKServeDescribeInferenceReq(String[] segments) {
        return "models".equals(segments[2]) && "infer".equals(segments[segments.length - 1]);
    }

    private void handleKServeDescribeModel(ChannelHandlerContext ctx, String[] segments)
            throws ModelNotFoundException {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 4) {
            modelVersion = segments[5];
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        Endpoint endpoint = endpoints.get(modelName);
        List<Workflow> workflows;
        if (endpoint != null) {
            workflows = endpoint.getWorkflows();
        } else {
            workflows = Collections.emptyList();
        }

        // TODO: How to handle multiple models and version?
        KServeDescribeModelResponse response = new KServeDescribeModelResponse();
        response.versions = new ArrayList<>();
        Model model = null;
        for (Workflow wf : workflows) {
            String version = wf.getVersion();
            if (modelVersion != null && !modelVersion.equals(version)) {
                continue;
            }
            if (version != null) {
                // TODO: null is a valid version in DJL
                response.versions.add(version);
            }
            if (model != null) {
                // only add one model
                continue;
            }

            for (ModelInfo<Input, Output> modelInfo : wf.getModels()) {
                if (modelInfo.getStatus() == ModelInfo.Status.READY) {
                    response.name = wf.getName();
                    response.setPlatform(modelInfo.getEngineName());
                    Device device = modelInfo.getModels().keySet().iterator().next();
                    model = modelInfo.getModel(device);

                    DataType dataType = model.getDataType();
                    if (model.describeInput() != null) {
                        for (Pair<String, Shape> input : model.describeInput()) {
                            response.addInput(input.getKey(), dataType, input.getValue());
                        }
                    }

                    if (model.describeOutput() != null) {
                        for (Pair<String, Shape> output : model.describeOutput()) {
                            response.addOutput(output.getKey(), dataType, output.getValue());
                        }
                    }
                    break;
                }
            }
        }
        if (model == null) {
            throw new ModelNotFoundException(
                    "Model not found: "
                            + modelName
                            + (modelVersion == null ? "" : '/' + modelVersion));
        }

        NettyUtils.sendJsonResponse(ctx, response);
    }

    private void handleKServeDescribeHealth(ChannelHandlerContext ctx) {
        ModelManager.getInstance()
                .workerStatus()
                .thenAccept(
                        w -> {
                            boolean hasFailure = (boolean) w.get("hasFailure");
                            boolean hasPending = (boolean) w.get("hasPending");

                            HttpResponseStatus httpResponseStatus;
                            if (hasFailure || hasPending) {
                                httpResponseStatus = HttpResponseStatus.FAILED_DEPENDENCY;
                            } else {
                                httpResponseStatus = HttpResponseStatus.OK;
                            }
                            NettyUtils.sendJsonResponse(ctx, EMPTY_BODY, httpResponseStatus);
                        });
    }

    private void handleKServeDescribeModelReady(ChannelHandlerContext ctx, String[] segments)
            throws ModelNotFoundException {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 5) {
            modelVersion = segments[5];
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(modelName, modelVersion);
        ModelInfo.Status status = modelInfo.getStatus();

        HttpResponseStatus httpResponseStatus;
        if (status == ModelInfo.Status.READY) {
            httpResponseStatus = HttpResponseStatus.OK;
        } else {
            httpResponseStatus = HttpResponseStatus.FAILED_DEPENDENCY;
        }
        NettyUtils.sendJsonResponse(ctx, EMPTY_BODY, httpResponseStatus);
    }

    private ModelInfo<Input, Output> getModelInfo(String modelName, String modelVersion)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);
        Collection<ModelInfo<Input, Output>> models;
        if (workflow != null) {
            models = workflow.getModels();
        } else {
            models = Collections.emptyList();
        }
        if (models.isEmpty()) {
            throw new ModelNotFoundException(
                    "Model not found: "
                            + modelName
                            + (modelVersion == null ? "" : '/' + modelVersion));
        }
        return models.iterator().next();
    }

    private void inference(ChannelHandlerContext ctx, FullHttpRequest req, String[] segments)
            throws ModelNotFoundException, IOException {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 5) {
            modelVersion = segments[5];
        }

        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);
        if (workflow == null) {
            throw new ModelNotFoundException("Parameter model_url is required.");
        }

        try (Reader reader =
                new InputStreamReader(
                        new ByteBufInputStream(req.content()), StandardCharsets.UTF_8)) {
            InferenceRequest request = JsonUtils.GSON.fromJson(reader, InferenceRequest.class);
            Input input = request.toInput();

            InferenceResponse response = new InferenceResponse(request.id, modelName, modelVersion);

            modelManager
                    .runJob(workflow, input)
                    .whenComplete(
                            (o, t) -> {
                                if (o != null) {
                                    responseOutput(response, o, ctx);
                                }
                            })
                    .exceptionally(
                            t -> {
                                onException((Exception) t, ctx);
                                return null;
                            });
        }
    }

    private void responseOutput(
            InferenceResponse response, Output output, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        int code = output.getCode();
        if (code == 200) {
            status = HttpResponseStatus.OK;

            byte[] data = output.getAsBytes(0);
            try (NDManager manager = NDManager.newBaseManager()) {
                NDList list = NDList.decode(manager, data);
                response.outputs = new KServeTensor[list.size()];
                for (int i = 0; i < response.outputs.length; ++i) {
                    response.outputs[i] = KServeTensor.fromTensor(list.get(i));
                }
            }
        } else {
            if (code >= 500) {
                status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
            } else if (code >= 400) {
                status = HttpResponseStatus.BAD_REQUEST;
            } else {
                status = new HttpResponseStatus(code, output.getMessage());
            }
        }

        NettyUtils.sendJsonResponse(ctx, response, status);
    }

    private void onException(Exception ex, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (ex instanceof ModelNotFoundException) {
            status = HttpResponseStatus.NOT_FOUND;
        } else if (ex instanceof MethodNotAllowedException) {
            status = HttpResponseStatus.METHOD_NOT_ALLOWED;
        } else if (ex instanceof TranslateException) {
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (ex instanceof WlmException) {
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", ex);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        Map<String, String> content = new ConcurrentHashMap<>();
        content.put("error", ex.getMessage());

        NettyUtils.sendJsonResponse(ctx, content, status);
    }

    private static final class InferenceRequest {

        String id;
        KServeTensor[] inputs;

        Input toInput() {
            Input input = new Input();
            try (NDManager manager = NDManager.newBaseManager();
                    NDList list = new NDList()) {
                for (KServeTensor tensor : inputs) {
                    list.add(tensor.toTensor(manager));
                }
                // here must be getAsBytes because list will die after the lifecycle
                input.add("data", list.getAsBytes());
            }
            return input;
        }
    }

    private static final class InferenceResponse {

        @SerializedName("model_name")
        String modelName;

        @SerializedName("model_version")
        String modelVersion;

        String id;
        KServeTensor[] outputs;

        public InferenceResponse(String id, String modelName, String modelVersion) {
            this.id = id;
            this.modelName = modelName;
            this.modelVersion = modelVersion;
        }

        public String getModelName() {
            return modelName;
        }

        public String getModelVersion() {
            return modelVersion;
        }

        public String getId() {
            return id;
        }
    }
}
