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

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.ModelNotFoundException;
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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Pattern PATTERN = Pattern.compile("^/v2/.+");

    private static final Logger logger = LoggerFactory.getLogger(KServeRequestHandler.class);

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
            try {
                handleKServeDescribeModel(ctx, segments);
            } catch (Exception exception) {
                onException(exception, ctx);
            }
        } else {
            throw new ResourceNotFoundException();
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments, HttpMethod method) {
        return HttpMethod.GET.equals(method) && segments.length == 4
                || (segments.length == 6 && "version".equals(segments[4]))
                        && "v2".equals(segments[1])
                        && "models".equals(segments[2]);
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
        if (endpoint == null) {
            throw new ModelNotFoundException(
                    "Model not found for the given model: "
                            + modelName
                            + " and model version "
                            + modelVersion);
        }

        List<Workflow> workflows = endpoint.getWorkflows();
        if (workflows.isEmpty()) {
            throw new ModelNotFoundException(
                    "Model not found for the given model: "
                            + modelName
                            + " and model version "
                            + modelVersion);
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

        NettyUtils.sendJsonResponse(ctx, response);
    }

    void onException(Exception ex, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (ex instanceof ModelNotFoundException) {
            status = HttpResponseStatus.NOT_FOUND;
        } else {
            logger.warn("Unexpected error", ex);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        Map<String, String> content = new ConcurrentHashMap<>();
        content.put("error", ex.getMessage());

        NettyUtils.sendJsonResponse(ctx, content, status);
    }
}
