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
import ai.djl.metric.Metric;
import ai.djl.modality.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.cache.CacheManager;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.translate.TranslateException;

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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests for the management API. */
public class InferenceRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(InferenceRequestHandler.class);
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");
    private static final Metric RESPONSE_2_XX = new Metric("2XX", 1);
    private static final Metric RESPONSE_4_XX = new Metric("4XX", 1);
    private static final Metric RESPONSE_5_XX = new Metric("5XX", 1);
    private static final Metric WLM_ERROR = new Metric("WlmError", 1);
    private static final Metric SERVER_ERROR = new Metric("ServerError", 1);
    private static final Pattern PATTERN =
            Pattern.compile("/(ping|invocations|predictions)([/?].*)?|/models/.+/invoke");

    private static final String X_SYNCHRONOUS = "x-synchronous";
    private static final String X_STARTING_TOKEN = "x-starting-token";
    private static final String X_NEXT_TOKEN = "x-next-token";
    private static final String X_MAX_ITEMS = "x-max-items";

    private RequestParser requestParser;

    /** default constructor. */
    public InferenceRequestHandler() {
        this.requestParser = new RequestParser();
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
            Set<String> startModels = ModelManager.getInstance().getStartupWorkflows();
            if (startModels.size() == 1) {
                modelName = startModels.iterator().next();
            }
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
                            Input.class,
                            Output.class,
                            -1,
                            -1,
                            -1,
                            -1);
            Workflow wf = new Workflow(modelInfo);

            modelManager
                    .registerWorkflow(wf)
                    .thenApply(p -> modelManager.initWorkers(wf, deviceName, -1, -1))
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
                .whenCompleteAsync(
                        (o, t) -> {
                            if (o != null) {
                                String sync = input.getProperty(X_SYNCHRONOUS, "true");
                                if (Boolean.parseBoolean(sync)) {
                                    sendOutput(o, ctx);
                                    return;
                                }

                                CacheManager cm = CacheManager.getInstance();
                                String nextToken = cm.put(o);
                                Output out = new Output();
                                out.setCode(o.getCode());
                                out.setMessage(o.getMessage());
                                out.getProperties().putAll(out.getProperties());
                                out.addProperty(X_NEXT_TOKEN, nextToken);
                                sendOutput(out, ctx);
                            }
                        })
                .exceptionally(
                        t -> {
                            onException(t.getCause(), ctx);
                            return null;
                        });
    }

    private void getCacheResult(ChannelHandlerContext ctx, Input input, String startingToken) {
        int limit = Integer.parseInt(input.getProperty(X_MAX_ITEMS, "-1"));
        if (limit < 0) {
            limit = Integer.MAX_VALUE;
        }

        CacheManager cm = CacheManager.getInstance();
        Output output = cm.get(startingToken);
        if (output == null) {
            throw new BadRequestException("Invalid " + X_STARTING_TOKEN);
        }
        BytesSupplier data = output.getData();
        if (!(data instanceof ChunkedBytesSupplier)) {
            logger.warn("Output doesn't support async response");
            sendOutput(output, ctx);
            return;
        }

        ChunkedBytesSupplier cbs = (ChunkedBytesSupplier) output.getData();
        List<byte[]> list = new ArrayList<>();
        int size = 0;
        for (int i = 0; i < limit; ++i) {
            byte[] buf = cbs.poll();
            if (buf == null) {
                break;
            }
            size += buf.length;
            list.add(buf);
        }
        byte[] buf = new byte[size];
        int pos = 0;
        for (byte[] array : list) {
            System.arraycopy(array, 0, buf, pos, array.length);
            pos += array.length;
        }
        Output o = new Output();
        o.setCode(output.getCode());
        o.setMessage(output.getMessage());
        o.getProperties().putAll(output.getProperties());
        o.add(buf);
        if (cbs.hasNext()) {
            o.addProperty(X_NEXT_TOKEN, startingToken);
        } else {
            // clean up cache
            cm.remove(startingToken);
        }
        sendOutput(o, ctx);
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
            HttpResponse resp = new DefaultHttpResponse(HttpVersion.HTTP_1_1, status, true);
            for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
                resp.headers().set(entry.getKey(), entry.getValue());
            }
            NettyUtils.sendHttpResponse(ctx, resp, true);
            ChunkedBytesSupplier supplier = (ChunkedBytesSupplier) data;
            try {
                while (supplier.hasNext()) {
                    byte[] buf = supplier.nextChunk(1, TimeUnit.MINUTES);
                    ByteBuf bb = Unpooled.wrappedBuffer(buf);
                    ctx.writeAndFlush(new DefaultHttpContent(bb));
                }
                ctx.writeAndFlush(LastHttpContent.EMPTY_LAST_CONTENT);
            } catch (InterruptedException | IllegalStateException e) {
                logger.warn("Chunk reading interrupted", e);
                ctx.newFailedFuture(e);
            }
            return;
        }

        FullHttpResponse resp = new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, true);
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
