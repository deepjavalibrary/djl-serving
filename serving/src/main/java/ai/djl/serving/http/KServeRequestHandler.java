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
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.Pair;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the KServe API. */
public class KServeRequestHandler extends HttpRequestHandler {

    private static final Pattern PATTERN = Pattern.compile("^/v2([/?].*)?");
    private static final String[] TYPES = {"FOR_KSERVE", "FOR_DJL_SERVING"};
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
        for (String segment : segments) {
            System.out.println(segment);
        }
        System.out.println("segments.length" + segments.length);
        HttpMethod method = req.method();
        if (HttpMethod.GET.equals(method) && isKServeDescribeModelReq(segments, method)) {
            handleKServeDescribeModel(ctx, segments);
        } else if (isKServeDescribeHealthReq(segments, method)) {
            handleKServeDescribeHealth(ctx, segments);
        }
    }

    private boolean isKServeDescribeModelReq(String[] segments, HttpMethod method) {
        System.out.println("This is a models request");
        return HttpMethod.GET.equals(method)
                && (segments.length == 4 || (segments.length == 6 && "version".equals(segments[4])))
                && "v2".equals(segments[1])
                && "models".equals(segments[2]);
    }

    private boolean isKServeDescribeHealthReq(String[] segments, HttpMethod method) {
        System.out.println("This is a health request");
        return "v2".equals(segments[1])
                && HttpMethod.GET.equals(method)
                && ((segments.length == 4 && "health".equals(segments[2]))
                        || ("models".equals(segments[2])
                                && "ready".equals(segments[segments.length - 1])));
    }

    private void handleKServeDescribeModel(ChannelHandlerContext ctx, String[] segments) {
        System.out.println("handle model req");
        String modelName = segments[3];
        String modelVersion = null;
        if (segments.length > 4) {
            modelVersion = segments[5];
        }
        ModelManager modelManager = ModelManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(modelName, modelVersion, false);

        // TODO: Search all workflows if model is not found here.
        ModelInfo<Input, Output> modelInfo =
                workflow.getModels().stream()
                        .filter(
                                model ->
                                        modelName.equals(
                                                model.getModel(model.withDefaultDevice(null))
                                                        .getName()))
                        .findAny()
                        .get();

        KServeDescribeModelResponse response = new KServeDescribeModelResponse();
        // TODO: Include all versions
        response.addModelVersion(modelInfo.getVersion());

        response.setPlatformForEngineName(modelInfo.getEngineName());

        ZooModel<Input, Output> model = modelInfo.getModel(modelInfo.withDefaultDevice(null));
        response.setName(model.getName());
        DataType dataType = model.getDataType();

        for (Pair<String, Shape> input : model.describeInput()) {
            response.addInput(input.getKey(), dataType, input.getValue());
        }

        if (null != model.describeOutput()) {
            for (Pair<String, Shape> output : model.describeOutput()) {
                response.addOutput(output.getKey(), dataType, output.getValue());
            }
        }

        NettyUtils.sendJsonResponse(ctx, response);
    }

    private void handleKServeDescribeHealth(ChannelHandlerContext ctx, String[] segments) {
        if ("models".equals(segments[2])) {
            String modelName = segments[3];
            String modelVersion = null;
            if (segments.length > 5) {
                modelVersion = segments[5];
            }
            ModelManager modelManager = ModelManager.getInstance();
            modelManager
                    .modelStatu(modelName, modelVersion)
                    .thenAccept(r -> NettyUtils.sendHttpResponse(ctx, r, true));
        } else {
            switch (segments[3]) {
                case "ready":
                case "live":
                    ModelManager.getInstance()
                            .workerStatus(TYPES[0])
                            .thenAccept(r -> NettyUtils.sendHttpResponse(ctx, r, true));
                    break;
            }
        }
    }
}
