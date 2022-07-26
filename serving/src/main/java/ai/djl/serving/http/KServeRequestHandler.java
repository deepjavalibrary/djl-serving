/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.Pair;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Pattern PATTERN = Pattern.compile("^/v2([/?].*)?");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        HttpMethod method = req.method();

        if (HttpMethod.GET.equals(method) && isKServeDescribeModelReq(segments, method)) {
            handleKServeDescribeModel(ctx, segments);
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments, HttpMethod method) {
        return HttpMethod.GET.equals(method) && segments.length == 4
                || (segments.length == 6 && "version".equals(segments[4]))
                        && "v2".equals(segments[1])
                        && "models".equals(segments[2]);
    }

    private void handleKServeDescribeModel(ChannelHandlerContext ctx, String[] segments) {
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 4) {
            modelVersion = segments[5];
        }

        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        Endpoint endpoint = endpoints.get(modelName);
        if (endpoint == null) {
            sendModelNotFoundError(modelName, modelVersion, ctx);
            return;
        }

        List<Workflow> workflows = endpoint.getWorkflows();
        if (workflows.isEmpty()) {
            sendModelNotFoundError(modelName, modelVersion, ctx);
            return;
        }

        List<ModelInfo<Input, Output>> models =
                workflows.stream()
                        .flatMap(w -> w.getModels().stream())
                        .collect(Collectors.toList());

        List<String> versions =
                models.stream()
                        .map(ModelInfo::getVersion)
                        .filter(Objects::nonNull)
                        .collect(Collectors.toList());

        ModelInfo<Input, Output> modelInfo = models.get(0);

        KServeDescribeModelResponse response = new KServeDescribeModelResponse();
        response.setVersions(versions);
        response.setPlatformForEngineName(modelInfo.getEngineName());

        ZooModel<Input, Output> model = modelInfo.getModel(modelInfo.withDefaultDevice(null));
        response.setName(model.getName());
        DataType dataType = model.getDataType();

        if (model.describeInput() == null || model.describeOutput() == null) {
            String errorMsg =
                    "Input/Output shapes are unknown, "
                            + "please run predict or forward once and call describe model again";

            Map<String, String> content = new ConcurrentHashMap<>();
            content.put("error", errorMsg);

            NettyUtils.sendJsonResponse(ctx, content, HttpResponseStatus.NOT_FOUND);
            return;
        }

        for (Pair<String, Shape> input : model.describeInput()) {
            response.addInput(input.getKey(), dataType, input.getValue());
        }

        for (Pair<String, Shape> output : model.describeOutput()) {
            response.addOutput(output.getKey(), dataType, output.getValue());
        }

        NettyUtils.sendJsonResponse(ctx, response);
    }

    private void sendModelNotFoundError(
            String modelName, String modelVersion, ChannelHandlerContext ctx) {
        String errorMsg =
                "Model not found for the given modelname "
                        + modelName
                        + " and model version "
                        + modelVersion;
        Map<String, String> content = new ConcurrentHashMap<>();
        content.put("error", errorMsg);

        NettyUtils.sendJsonResponse(ctx, content, HttpResponseStatus.NOT_FOUND);
    }
}
