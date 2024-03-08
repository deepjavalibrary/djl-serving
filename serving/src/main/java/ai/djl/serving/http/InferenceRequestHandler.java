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
import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.inference.streaming.PublisherBytesSupplier;
import ai.djl.metric.Metric;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.cache.CacheEngine;
import ai.djl.serving.cache.CacheManager;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;

import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.DefaultHttpContent;
import io.netty.handler.codec.http.DefaultHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.LastHttpContent;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the management API. */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);

    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");
    private static final Metric RESPONSE_2_XX = new Metric("Response_2XX", 1);
    private static final Metric RESPONSE_4_XX = new Metric("Response_4XX", 1);
    private static final Metric RESPONSE_5_XX = new Metric("Response_5XX", 1);
    private static final Metric WLM_ERROR = new Metric("WlmError", 1);
    private static final Metric SERVER_ERROR = new Metric("ServerError", 1);
    private static final Pattern PATTERN =
            Pattern.compile("/(ping|invocations|predictions)([/?].*)?|/models/.+/invoke");

    private static final String X_SYNCHRONOUS = "x-synchronous";
    private static final String X_STARTING_TOKEN = "x-starting-token";
    private static final String X_NEXT_TOKEN = "x-next-token";
    private static final String X_MAX_ITEMS = "x-max-items";
    private static final String X_CUSTOM_ATTRIBUTES = "X-Amzn-SageMaker-Custom-Attributes";

    private RequestParser requestParser;
    private int chunkReadTime;

    /** default constructor. */
    public InferenceRequestHandler() {
        this.requestParser = new RequestParser();
        chunkReadTime = ConfigManager.getInstance().getChunkedReadTimeout();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            String uri = req.uri();
            return PATTERN.matcher(uri).matches();
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
                        .thenAccept(
                                w -> {
                                    boolean hasFailure = (boolean) w.get("hasFailure");
                                    boolean hasPending = (boolean) w.get("hasPending");

                                    HttpResponseStatus status;
                                    if (hasFailure) {
                                        logger.info(
                                                "PING FAILED: {}",
                                                JsonUtils.GSON.toJson(w.get("data")));
                                        status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
                                    } else if (hasPending) {
                                        if (ConfigManager.getInstance().allowsMultiStatus()) {
                                            status = HttpResponseStatus.MULTI_STATUS;
                                        } else {
                                            status = HttpResponseStatus.OK;
                                        }
                                    } else {
                                        status = HttpResponseStatus.OK;
                                    }
                                    NettyUtils.sendJsonResponse(ctx, w.get("data"), status);
                                });
                break;
            case "invocations":
                handleInvocations(ctx, req, decoder, null);
                break;
            case "models":
                handleInvocations(ctx, req, decoder, segments[2]);
                break;
            case "predictions":
                handlePredictions(ctx, req, decoder, segments);
                break;
            default:
                throw new AssertionError("Invalid request uri: " + req.uri());
        }
    }

    /** {@inheritDoc} */
    @Override
    public void channelInactive(ChannelHandlerContext ctx) throws Exception {
        Session session = NettyUtils.getSession(ctx.channel());
        if (session != null) {
            Input input = session.getInput();
            if (input != null) {
                input.setCancelled(true);
            }
        }
        super.channelInactive(ctx);
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
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String modelName)
            throws ModelNotFoundException {
        Input input = requestParser.parseRequest(req, decoder);
        if (modelName == null) {
            modelName = NettyUtils.getParameter(decoder, "model_name", null);
        }
        if ((modelName == null || modelName.isEmpty())) {
            modelName = input.getProperty("model_name", null);
            if (modelName == null) {
                modelName = input.getAsString("model_name");
            }
        }
        if (modelName == null) {
            modelName = ModelManager.getInstance().getSingleStartupWorkflow().orElse(null);
            if (modelName == null) {
                throw new BadRequestException("Parameter model_name is required.");
            }
        }
        String version = NettyUtils.getParameter(decoder, "model_version", null);
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
        String startingToken = input.getProperty(X_STARTING_TOKEN, null);
        if (startingToken != null && !HttpMethod.OPTIONS.equals(req.method())) {
            CompletableFuture.runAsync(() -> getCacheResult(ctx, input, startingToken))
                    .exceptionally(
                            t -> {
                                onException(t.getCause(), ctx);
                                return null;
                            });
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        ConfigManager config = ConfigManager.getInstance();
        Workflow workflow = modelManager.getWorkflow(workflowName, version, true);
        if (workflow == null) {
            String regex = config.getModelUrlPattern();
            if (regex == null) {
                throw new ModelNotFoundException("Model or workflow not found: " + workflowName);
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
            String deviceName = input.getProperty("device", null);

            logger.info("Loading model {} from: {}", workflowName, modelUrl);

            ModelInfo<Input, Output> modelInfo =
                    new ModelInfo<>(
                            workflowName,
                            modelUrl,
                            version,
                            engineName,
                            deviceName,
                            Input.class,
                            Output.class,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1,
                            -1);
            Workflow wf = new Workflow(modelInfo);

            modelManager
                    .registerWorkflow(wf)
                    .thenAccept(p -> runJob(modelManager, ctx, wf, input))
                    .exceptionally(
                            t -> {
                                logger.error("Failed register workflow", t);
                                NettyUtils.sendError(ctx, t.getCause());
                                return null;
                            });
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
        Session session = NettyUtils.getSession(ctx.channel());
        session.setInput(input);
        String sync = input.getProperty(X_SYNCHRONOUS, "true");
        if (Boolean.parseBoolean(sync)) { // Synchronous
            modelManager
                    .runJob(workflow, input)
                    .whenCompleteAsync(
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
        } else { // Asynchronous
            CacheEngine cache = CacheManager.getCacheEngine();
            String nextToken = cache.create(input);

            // Store pending message to be sent for unfinished computations
            Output pending = new Output();
            pending.setMessage("The model result is not yet available");
            pending.setCode(202);
            pending.addProperty(X_NEXT_TOKEN, nextToken);
            pending.addProperty(X_CUSTOM_ATTRIBUTES, X_NEXT_TOKEN + '=' + nextToken);
            cache.put(nextToken, pending)
                    .thenAccept(
                            ignored -> {
                                // Send back token to user
                                Output out = new Output();
                                out.addProperty(X_NEXT_TOKEN, nextToken);
                                out.addProperty(
                                        X_CUSTOM_ATTRIBUTES, X_NEXT_TOKEN + '=' + nextToken);
                                sendOutput(out, ctx);

                                // Run model
                                modelManager
                                        .runJob(workflow, input)
                                        .whenCompleteAsync(
                                                (o, t) -> {
                                                    if (o != null) {
                                                        cache.put(nextToken, o);
                                                    } else {
                                                        Output failOut = new Output();
                                                        failOut.setCode(500);
                                                        failOut.setMessage(t.getMessage());
                                                        cache.put(nextToken, failOut);
                                                    }
                                                });
                            });
        }
    }

    private void getCacheResult(ChannelHandlerContext ctx, Input input, String startingToken) {
        int limit = Integer.parseInt(input.getProperty(X_MAX_ITEMS, "-1"));
        if (limit < 0) {
            limit = Integer.MAX_VALUE;
        }

        CacheEngine cache = CacheManager.getCacheEngine();
        Output output;
        try {
            output = cache.get(startingToken, limit);
        } catch (RuntimeException e) {
            throw new BadRequestException("Failed to lookup cache element", e);
        }
        if (output == null) {
            throw new BadRequestException("Invalid " + X_STARTING_TOKEN + ": " + startingToken);
        }
        sendOutput(output, ctx);
    }

    void sendOutput(Output output, ChannelHandlerContext ctx) {
        /*
         * We can load the models based on the configuration file. Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx == null) {
            return;
        }

        HttpResponseStatus status;
        int code = output.getCode();
        if (code == 200) {
            status = HttpResponseStatus.OK;
            SERVER_METRIC.info("{}", RESPONSE_2_XX);
        } else {
            if (code >= 500) {
                SERVER_METRIC.info("{}", RESPONSE_5_XX);
            } else if (code >= 400) {
                SERVER_METRIC.info("{}", RESPONSE_4_XX);
            } else {
                SERVER_METRIC.info("{}", RESPONSE_2_XX);
            }
            status = new HttpResponseStatus(code, output.getMessage());
        }
        BytesSupplier data = output.getData();
        if (data instanceof ChunkedBytesSupplier) {
            try {
                boolean first = true;
                ChunkedBytesSupplier supplier = (ChunkedBytesSupplier) data;
                while (supplier.hasNext()) {
                    byte[] buf = supplier.nextChunk(chunkReadTime, TimeUnit.SECONDS);
                    // Defer sending HTTP header until first chunk received.
                    // This allows inference update HTTP code.
                    if (first) {
                        code = output.getCode();
                        status = new HttpResponseStatus(code, output.getMessage());
                        HttpResponse resp = new DefaultHttpResponse(HttpVersion.HTTP_1_1, status);
                        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
                            resp.headers().set(entry.getKey(), entry.getValue());
                        }
                        NettyUtils.sendHttpResponse(ctx, resp, true, false);
                        first = false;
                    }

                    ByteBuf bb = Unpooled.wrappedBuffer(buf);
                    ctx.writeAndFlush(new DefaultHttpContent(bb));
                }
                ctx.writeAndFlush(LastHttpContent.EMPTY_LAST_CONTENT);
            } catch (InterruptedException | IllegalStateException e) {
                logger.warn("Chunk reading interrupted", e);
                ctx.disconnect();
                ctx.newFailedFuture(e);
            }
            return;
        }
        if (data instanceof PublisherBytesSupplier) {
            HttpResponse resp = new DefaultHttpResponse(HttpVersion.HTTP_1_1, status);
            for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
                resp.headers().set(entry.getKey(), entry.getValue());
            }
            NettyUtils.sendHttpResponse(ctx, resp, true);
            PublisherBytesSupplier supplier = (PublisherBytesSupplier) data;
            supplier.subscribe(
                    buf -> {
                        if (buf == null) {
                            // End stream
                            ctx.writeAndFlush(LastHttpContent.EMPTY_LAST_CONTENT);
                        } else if (buf.length > 0) {
                            // Continue stream
                            ByteBuf bb = Unpooled.wrappedBuffer(buf);
                            ctx.writeAndFlush(new DefaultHttpContent(bb));
                        }
                    });
            return;
        }

        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status);
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            resp.headers().set(entry.getKey(), entry.getValue());
        }
        if (data != null) {
            resp.content().writeBytes(data.getAsBytes());
        }

        NettyUtils.sendHttpResponse(ctx, resp, true);
    }

    void onException(Throwable t, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (t instanceof TranslateException || t instanceof BadRequestException) {
            logger.debug(t.getMessage(), t);
            SERVER_METRIC.info("{}", RESPONSE_4_XX);
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmException) {
            logger.warn(t.getMessage(), t);
            SERVER_METRIC.info("{}", RESPONSE_5_XX);
            SERVER_METRIC.info("{}", WLM_ERROR);
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", t);
            SERVER_METRIC.info("{}", RESPONSE_5_XX);
            SERVER_METRIC.info("{}", SERVER_ERROR);
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
