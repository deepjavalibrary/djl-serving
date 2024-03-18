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
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.Adapter;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkerPool;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Pattern;

/** A class handling inbound HTTP requests to the management API for adapters. */
public class AdapterManagementRequestHandler extends HttpRequestHandler {

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

        if (segments.length < 4) {
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
                    handleListAdapters(ctx, modelName, decoder);
                    return;
                } else if (HttpMethod.POST.equals(method)) {
                    handleRegisterAdapter(ctx, modelName, decoder);
                    return;
                } else {
                    throw new MethodNotAllowedException();
                }
            }

            String adapterName = segments[2];
            if (HttpMethod.GET.equals(method)) {
                handleDescribeAdapter(ctx, modelName, adapterName);
            } else if (HttpMethod.DELETE.equals(method)) {
                handleUnregisterAdapter(ctx, modelName, adapterName);
            } else {
                throw new MethodNotAllowedException();
            }

        } else {
            // API /models/{modelName}/adapters/*

            String modelName = segments[2];
            if (segments.length < 5) {
                if (HttpMethod.GET.equals(method)) {
                    handleListAdapters(ctx, modelName, decoder);
                    return;
                } else if (HttpMethod.POST.equals(method)) {
                    handleRegisterAdapter(ctx, modelName, decoder);
                    return;
                } else {
                    throw new MethodNotAllowedException();
                }
            }

            String adapterName = segments[4];
            if (HttpMethod.GET.equals(method)) {
                handleDescribeAdapter(ctx, modelName, adapterName);
            } else if (HttpMethod.DELETE.equals(method)) {
                handleUnregisterAdapter(ctx, modelName, adapterName);
            } else {
                throw new MethodNotAllowedException();
            }
        }
    }

    private void handleListAdapters(
            ChannelHandlerContext ctx, String modelName, QueryStringDecoder decoder) {
        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException("The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);

        ListAdaptersResponse list = new ListAdaptersResponse();
        List<String> keys = new ArrayList<>(modelInfo.getAdapters().keySet());
        ListPagination pagination = new ListPagination(decoder, keys.size());
        if (pagination.getLast() < keys.size()) {
            list.setNextPageToken(String.valueOf(pagination.getLast()));
        }

        for (int i = pagination.getPageToken(); i < pagination.getLast(); ++i) {
            String adapterName = keys.get(i);
            Adapter adapter = modelInfo.getAdapter(adapterName);
            list.addAdapter(adapter.getName(), adapter.getSrc());
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleRegisterAdapter(
            ChannelHandlerContext ctx, String modelName, QueryStringDecoder decoder) {

        String adapterName = NettyUtils.getRequiredParameter(decoder, "name");
        String src = NettyUtils.getRequiredParameter(decoder, "src");

        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException("The model " + modelName + " was not found");
        }
        Adapter adapter = Adapter.newInstance(wp.getWpc(), adapterName, src);
        adapter.register(wp);

        String msg = "Adapter " + adapterName + " registered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleDescribeAdapter(
            ChannelHandlerContext ctx, String modelName, String adapterName) {
        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException("The model " + modelName + " was not found");
        }
        ModelInfo<Input, Output> modelInfo = getModelInfo(wp);
        Adapter adapter = modelInfo.getAdapter(adapterName);

        if (adapter == null) {
            throw new BadRequestException("The adapter " + adapterName + " was not found");
        }

        DescribeAdapterResponse adapterResponse = new DescribeAdapterResponse(adapter);
        NettyUtils.sendJsonResponse(ctx, adapterResponse);
    }

    private void handleUnregisterAdapter(
            ChannelHandlerContext ctx, String modelName, String adapterName) {

        WorkerPool<Input, Output> wp =
                ModelManager.getInstance().getWorkLoadManager().getWorkerPoolById(modelName);
        if (wp == null) {
            throw new BadRequestException("The model " + modelName + " was not found");
        }
        Adapter.unregister(wp, adapterName);

        String msg = "Adapter " + adapterName + " registered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private ModelInfo<Input, Output> getModelInfo(WorkerPool<Input, Output> wp) {
        if (!(wp.getWpc() instanceof ModelInfo)) {
            String modelName = wp.getWpc().getId();
            throw new BadRequestException("The worker " + modelName + " is not a model");
        }
        return (ModelInfo<Input, Output>) wp.getWpc();
    }
}
