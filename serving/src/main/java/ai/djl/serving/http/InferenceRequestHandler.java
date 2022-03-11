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
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;
import java.util.Map;
import java.util.Set;
import java.util.regex.Pattern;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class handling inbound HTTP requests for the management API. */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    private RequestParser requestParser;

    private static final Pattern PATTERN =
            Pattern.compile("^/(ping|invocations|predictions)([/?].*)?");

    /** default constructor. */
    public InferenceRequestHandler() {
        this.requestParser = new RequestParser();
    }

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
            String[] segments)
            throws ModelException {
        switch (segments[1]) {
            case "ping":
                ModelManager.getInstance()
                        .workerStatus()
                        .thenAccept(r -> NettyUtils.sendHttpResponse(ctx, r, true));
                break;
            case "invocations":
                handleInvocations(ctx, req, decoder);
                break;
            case "predictions":
                handlePredictions(ctx, req, decoder, segments);
                break;
            default:
                throw new AssertionError("Invalid request uri: " + req.uri());
        }
    }

    private void handlePredictions(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelNotFoundException {
        if (segments.length < 3) {
            throw new ResourceNotFoundException();
        }
        String modelName = segments[2];
        String version;
        if (segments.length > 3) {
            version = segments[3].isEmpty() ? null : segments[3];
        } else {
            version = null;
        }
        Input input = requestParser.parseRequest(req, decoder);
        predict(ctx, req, input, modelName, version);
    }

    private void handleInvocations(
            ChannelHandlerContext ctx, FullHttpRequest req, QueryStringDecoder decoder)
            throws ModelNotFoundException {
        Input input = requestParser.parseRequest(req, decoder);
        String modelName = NettyUtils.getParameter(decoder, "model_name", null);
        String version = NettyUtils.getParameter(decoder, "model_version", null);
        if ((modelName == null || modelName.isEmpty())) {
            modelName = input.getProperty("model_name", null);
            if (modelName == null) {
                modelName = input.getAsString("model_name");
            }
        }
        if (modelName == null) {
            Set<String> startModels = ModelManager.getInstance().getStartupWorkflows();
            if (startModels.size() == 1) {
                modelName = startModels.iterator().next();
            }
            if (modelName == null) {
                throw new BadRequestException("Parameter model_name is required.");
            }
        }
        if (version == null) {
            version = input.getProperty("model_version", null);
        }
        predict(ctx, req, input, modelName, version);
    }

    private void predict(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            Input input,
            String workflowName,
            String version)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        ConfigManager config = ConfigManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(workflowName, version, true);
        if (workflow == null) {
            String regex = config.getModelUrlPattern();
            if (regex == null) {
                throw new ModelNotFoundException("Model not found: " + workflowName);
            }
            String modelUrl = input.getProperty("model_url", null);
            if (modelUrl == null) {
                modelUrl = input.getAsString("model_url");
                if (modelUrl == null) {
                    throw new ModelNotFoundException("Parameter model_url is required.");
                }
                if (!modelUrl.matches(regex)) {
                    throw new ModelNotFoundException("Permission denied: " + modelUrl);
                }
            }
            String engineName = input.getProperty("engine_name", null);
            String deviceName = input.getProperty("device", "-1");

            logger.info("Loading model {} from: {}", workflowName, modelUrl);
            ModelInfo modelInfo =
                    new ModelInfo(
                            workflowName,
                            modelUrl,
                            version,
                            engineName,
                            config.getJobQueueSize(),
                            config.getMaxIdleTime(),
                            config.getMaxBatchDelay(),
                            config.getBatchSize());
            Workflow wf = new Workflow(modelInfo);

            modelManager
                    .registerWorkflow(wf, deviceName)
                    .thenApply(p -> modelManager.scaleWorkers(wf, deviceName, 1, -1))
                    .thenAccept(p -> runJob(modelManager, ctx, p, input));
            return;
        }

        if (HttpMethod.OPTIONS.equals(req.method())) {
            NettyUtils.sendJsonResponse(ctx, "{}");
            return;
        }

        runJob(modelManager, ctx, workflow, input);
    }

    void runJob(
            ModelManager modelManager, ChannelHandlerContext ctx, Workflow workflow, Input input) {
        modelManager
                .runJob(workflow, input)
                .whenComplete(
                        (o, t) -> {
                            if (o != null) {
                                sendOutput(o, ctx);
                            }
                        })
                .exceptionally(
                        t -> {
                            onException(t.getCause(), ctx);
                            return null;
                        });
    }

    void sendOutput(Output output, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (output.getCode() == 200) {
            status = HttpResponseStatus.OK;
        } else {
            status = new HttpResponseStatus(output.getCode(), output.getMessage());
        }
        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            resp.headers().set(entry.getKey(), entry.getValue());
        }
        BytesSupplier data = output.getData();
        if (data != null) {
            resp.content().writeBytes(data.getAsBytes());
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
    }

    void onException(Throwable t, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (t instanceof TranslateException) {
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmException) {
            logger.warn(t.getMessage(), t);
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", t);
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendError(ctx, status, t);
        }
    }
}
