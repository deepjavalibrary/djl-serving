/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.serving.http.list.ListAdaptersResponse;
import ai.djl.serving.http.list.ListPagination;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.Adapter;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPool;
import ai.djl.serving.wlm.util.WlmCapacityException;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.translate.TranslateException;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests to the management API for adapters. */
public class AdapterManagementRequestHandler extends HttpRequestHandler {

    private static final Logger logger =
            LoggerFactory.getLogger(AdapterManagementRequestHandler.class);

    static final Pattern ADAPTERS_PATTERN =
            Pattern.compile("^(/models/[^/^?]+)?/adapters([/?].*)?");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            String uri = req.uri();
            return ADAPTERS_PATTERN.matcher(uri).matches();
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
        HttpMethod method = req.method();

        if ("adapters".equals(segments[1])) {
            // API /adapters/*
            String modelName =
                    ModelManager.getInstance()
                            .getSingleStartupWorkflow()
                            .orElseThrow(
                                    () ->
                                            new BadRequestException(
                                                    "The adapter must be prefixed with a model"
                                                        + " unless there is only a single startup"
                                                        + " model used."));
            if (segments.length < 3) {
                if (HttpMethod.GET.equals(method)) {
                    handleListAdapters(ctx, decoder, modelName);
                    return;
                } else if (HttpMethod.POST.equals(method)) {
                    handleRegisterAdapter(ctx, decoder, modelName);
                    return;
                } else {
                    throw new MethodNotAllowedException();
                }
            }

            String adapterName = segments[2];
            if (HttpMethod.GET.equals(method)) {
                handleDescribeAdapter(ctx, modelName, adapterName);
            } else if (HttpMethod.POST.equals(method) && "update".equalsIgnoreCase(segments[3])) {
                handleUpdateAdapter(ctx, decoder, modelName, adapterName);
            } else if (HttpMethod.DELETE.equals(method)) {
                handleUnregisterAdapter(ctx, modelName, adapterName);
            } else {
                throw new MethodNotAllowedException();
            }
        } else if ("models".equals(segments[1])) {
            // API /models/{model_name}/adapters/*

            String modelName = segments[2];
            if (segments.length < 5) {
                if (HttpMethod.GET.equals(method)) {
                    handleListAdapters(ctx, decoder, modelName);
                    return;
                } else if (HttpMethod.POST.equals(method)) {
                    handleRegisterAdapter(ctx, decoder, modelName);
                    return;
                } else {
                    throw new MethodNotAllowedException();
                }
            }

            String adapterName = segments[4];
            if (HttpMethod.GET.equals(method)) {
                handleDescribeAdapter(ctx, modelName, adapterName);
            } else if (HttpMethod.POST.equals(method) && "update".equalsIgnoreCase(segments[5])) {
                handleUpdateAdapter(ctx, decoder, modelName, adapterName);
            } else if (HttpMethod.DELETE.equals(method)) {
                handleUnregisterAdapter(ctx, modelName, adapterName);
            } else {
                throw new MethodNotAllowedException();
            }
        }
    }

    private void handleListAdapters(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName) {
        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException(404, "The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        boolean enableLora =
                Boolean.parseBoolean(
                        modelInfo.getProperties().getProperty("option.enable_lora", "false"));
        if (!enableLora) {
            throw new BadRequestException("LoRA is not enabled.");
        }

        ListAdaptersResponse list = new ListAdaptersResponse();
        List<String> keys = new ArrayList<>(modelInfo.getAdapters().keySet());
        ListPagination pagination = new ListPagination(decoder, keys.size());
        if (pagination.getLast() < keys.size()) {
            list.setNextPageToken(String.valueOf(pagination.getLast()));
        }

        for (int i = pagination.getPageToken(); i < pagination.getLast(); ++i) {
            String adapterName = keys.get(i);
            Adapter<Input, Output> adapter = modelInfo.getAdapter(adapterName);
            list.addAdapter(adapter.getName(), adapter.getSrc(), adapter.isPin());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleRegisterAdapter(
            ChannelHandlerContext ctx, QueryStringDecoder decoder, String modelName) {

        String adapterName = NettyUtils.getRequiredParameter(decoder, "name");
        String src = NettyUtils.getRequiredParameter(decoder, "src");

        WorkLoadManager wlm = ModelManager.getInstance().getWorkLoadManager();
        WorkerPool<Input, Output> wp = wlm.getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException(404, "The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        boolean enableLora =
                Boolean.parseBoolean(
                        modelInfo.getProperties().getProperty("option.enable_lora", "false"));
        if (!enableLora) {
            throw new BadRequestException("LoRA is not enabled.");
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
            if (entry.getValue().size() == 1) {
                options.put(entry.getKey(), entry.getValue().get(0));
            }
        }
        boolean pin = Boolean.parseBoolean(options.getOrDefault("pin", "false"));
        Adapter<Input, Output> adapter =
                Adapter.newInstance(modelInfo, adapterName, src, pin, options);
        adapter.register(wlm)
                .whenCompleteAsync(
                        (o, t) -> {
                            if (o != null) {
                                sendOutput(o, ctx);
                            }
                        })
                .exceptionally(
                        t -> {
                            onException(t.getCause(), ctx);
                            Adapter.unregister(adapterName, modelInfo, wlm);
                            return null;
                        });
    }

    private void handleUpdateAdapter(
            ChannelHandlerContext ctx,
            QueryStringDecoder decoder,
            String modelName,
            String adapterName) {
        WorkLoadManager wlm = ModelManager.getInstance().getWorkLoadManager();
        WorkerPool<Input, Output> wp = wlm.getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException(404, "The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        boolean enableLora =
                Boolean.parseBoolean(
                        modelInfo.getProperties().getProperty("option.enable_lora", "false"));
        if (!enableLora) {
            throw new BadRequestException("LoRA is not enabled.");
        }

        Adapter<Input, Output> adapter = modelInfo.getAdapter(adapterName);

        if (adapter == null) {
            throw new BadRequestException(404, "The adapter " + adapterName + " was not found");
        }

        Map<String, String> options = new ConcurrentHashMap<>();
        for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
            if (entry.getValue().size() == 1) {
                options.put(entry.getKey(), entry.getValue().get(0));
            }
        }
        String src = options.get("src");
        boolean pin = Boolean.parseBoolean(options.getOrDefault("pin", "false"));

        if (src != null) {
            adapter.setSrc(src);
        }
        adapter.setPin(pin);

        adapter.update(wlm)
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
    }

    private void handleDescribeAdapter(
            ChannelHandlerContext ctx, String modelName, String adapterName) {
        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException(404, "The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        boolean enableLora =
                Boolean.parseBoolean(
                        modelInfo.getProperties().getProperty("option.enable_lora", "false"));
        if (!enableLora) {
            throw new BadRequestException("LoRA is not enabled.");
        }

        Adapter<Input, Output> adapter = modelInfo.getAdapter(adapterName);

        if (adapter == null) {
            throw new BadRequestException(404, "The adapter " + adapterName + " was not found");
        }

        DescribeAdapterResponse adapterResponse = new DescribeAdapterResponse(adapter);
        NettyUtils.sendJsonResponse(ctx, adapterResponse);
    }

    private void handleUnregisterAdapter(
            ChannelHandlerContext ctx, String modelName, String adapterName) {
        WorkLoadManager wlm = ModelManager.getInstance().getWorkLoadManager();
        WorkerPool<Input, Output> wp = wlm.getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException(404, "The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        boolean enableLora =
                Boolean.parseBoolean(
                        modelInfo.getProperties().getProperty("option.enable_lora", "false"));
        if (!enableLora) {
            throw new BadRequestException("LoRA is not enabled.");
        }

        Adapter.unregister(adapterName, modelInfo, wlm)
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
    }

    private ModelInfo<Input, Output> getModelInfo(WorkerPool<Input, Output> wp) {
        if (!(wp.getWpc() instanceof ModelInfo)) {
            String modelName = wp.getWpc().getId();
            throw new BadRequestException("The worker " + modelName + " is not a model");
        }
        return (ModelInfo<Input, Output>) wp.getWpc();
    }

    private void sendOutput(Output output, ChannelHandlerContext ctx) {
        if (ctx == null) {
            return;
        }

        int code = output.getCode();
        if (code >= 300) {
            NettyUtils.sendJsonResponse(
                    ctx,
                    new ErrorResponse(code, null, output.getMessage()),
                    HttpResponseStatus.valueOf(code));
        } else {
            NettyUtils.sendJsonResponse(
                    ctx, new StatusResponse(output.getMessage()), HttpResponseStatus.valueOf(code));
        }
    }

    private void onException(Throwable t, ChannelHandlerContext ctx) {
        ConfigManager config = ConfigManager.getInstance();
        int code;
        String requestIdLogPrefix = "";
        if (ctx != null) {
            String requestId = NettyUtils.getRequestId(ctx.channel());
            requestIdLogPrefix = "RequestId=[" + requestId + "]: ";
        }
        if (t instanceof TranslateException) {
            logger.debug("{}{}", requestIdLogPrefix, t.getMessage(), t);
            code = config.getBadRequestErrorHttpCode();
        } else if (t instanceof BadRequestException) {
            code = ((BadRequestException) t).getCode();
        } else if (t instanceof WlmException) {
            logger.warn("{}{}", requestIdLogPrefix, t.getMessage(), t);
            if (t instanceof WlmCapacityException) {
                code = config.getThrottleErrorHttpCode();
            } else {
                code = config.getWlmErrorHttpCode();
            }
        } else {
            logger.warn("{} Unexpected error", requestIdLogPrefix, t);
            code = config.getServerErrorHttpCode();
        }
        HttpResponseStatus status = HttpResponseStatus.valueOf(code);

        if (ctx != null) {
            NettyUtils.sendError(ctx, status, t);
        }
    }
}
