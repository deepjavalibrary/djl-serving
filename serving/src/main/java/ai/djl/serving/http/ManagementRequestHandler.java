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
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;
import ai.djl.util.JsonUtils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.util.CharsetUtil;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.regex.Pattern;

/**
 * A class handling inbound HTTP requests to the management API.
 *
 * <p>This class
 */
public class ManagementRequestHandler extends HttpRequestHandler {

    private static final Pattern WORKFLOWS_PATTERN = Pattern.compile("^/workflows([/?].*)?");
    private static final Pattern MODELS_PATTERN = Pattern.compile("^/models([/?].*)?");
    private static final Pattern INVOKE_PATTERN = Pattern.compile("^/models/.+/invoke$");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            String uri = req.uri();
            if (WORKFLOWS_PATTERN.matcher(uri).matches()) {
                return true;
            } else if (MODELS_PATTERN.matcher(uri).matches()) {
                return req.method() != HttpMethod.POST || !INVOKE_PATTERN.matcher(uri).matches();
            }
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
        if (segments.length < 3) {
            if (HttpMethod.GET.equals(method)) {
                if (MODELS_PATTERN.matcher(req.uri()).matches()) {
                    handleListModels(ctx, decoder);
                } else {
                    handleListWorkflows(ctx, decoder);
                }
                return;
            } else if (HttpMethod.POST.equals(method)) {
                if (MODELS_PATTERN.matcher(req.uri()).matches()) {
                    handleRegisterModel(ctx, req, decoder);
                } else {
                    handleRegisterWorkflow(ctx, decoder);
                }
                return;
            }
            throw new MethodNotAllowedException();
        }

        String modelName = segments[2];
        String version = null;
        if (segments.length > 3) {
            version = segments[3];
        }

        if (HttpMethod.GET.equals(method)) {
            handleDescribeWorkflow(ctx, modelName, version);
        } else if (HttpMethod.PUT.equals(method)) {
            handleScaleWorkflow(ctx, decoder, modelName, version);
        } else if (HttpMethod.DELETE.equals(method)) {
            handleUnregisterWorkflow(ctx, modelName, version);
        } else {
            throw new MethodNotAllowedException();
        }
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        List<String> keys = new ArrayList<>(endpoints.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        ListPagination pagination = new ListPagination(decoder, keys.size());
        if (pagination.last < keys.size()) {
            list.setNextPageToken(String.valueOf(pagination.last));
        }

        for (int i = pagination.pageToken; i < pagination.last; ++i) {
            String workflowName = keys.get(i);
            for (Workflow workflow : endpoints.get(workflowName).getWorkflows()) {
                for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
                    String status = wpc.getStatus().toString();
                    String id = wpc.getId();
                    String name;
                    if (workflowName.equals(id)) {
                        name = workflowName;
                    } else {
                        name = workflowName + ':' + id;
                    }
                    list.addModel(name, workflow.getVersion(), wpc.getModelUrl(), status);
                }
            }
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleListWorkflows(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        List<String> keys = new ArrayList<>(endpoints.keySet());
        Collections.sort(keys);
        ListWorkflowsResponse list = new ListWorkflowsResponse();

        ListPagination pagination = new ListPagination(decoder, keys.size());
        if (pagination.last <= keys.size()) {
            list.setNextPageToken(String.valueOf(pagination.last));
        }

        for (int i = pagination.pageToken; i < pagination.last; ++i) {
            String workflowName = keys.get(i);
            for (Workflow w : endpoints.get(workflowName).getWorkflows()) {
                list.addWorkflow(workflowName, w.getVersion());
            }
        }

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleDescribeWorkflow(
            ChannelHandlerContext ctx, String workflowName, String version)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        DescribeWorkflowResponse[] resp = modelManager.describeWorkflow(workflowName, version);
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterModel(
            final ChannelHandlerContext ctx, FullHttpRequest request, QueryStringDecoder decoder) {
        LoadModelRequest req;
        CharSequence contentType = HttpUtil.getMimeType(request);
        if (HttpHeaderValues.APPLICATION_JSON.contentEqualsIgnoreCase(contentType)) {
            String body = request.content().toString(CharsetUtil.UTF_8);
            req = JsonUtils.GSON.fromJson(body, LoadModelRequest.class);
        } else {
            req = new LoadModelRequest(decoder);
        }

        Workflow workflow;
        URI uri = WorkflowDefinition.toWorkflowUri(req.getModelUrl());
        if (uri != null) {
            try {
                workflow = WorkflowDefinition.parse(req.getModelName(), uri).toWorkflow();
            } catch (IOException | BadWorkflowException e) {
                NettyUtils.sendError(ctx, e.getCause());
                return;
            }
        } else {
            ModelInfo<Input, Output> modelInfo =
                    new ModelInfo<>(
                            req.getModelName(),
                            req.getModelUrl(),
                            req.getVersion(),
                            req.getEngineName(),
                            req.getDeviceName(),
                            Input.class,
                            Output.class,
                            req.getJobQueueSize(),
                            req.getMaxIdleSeconds(),
                            req.getMaxBatchDelayMillis(),
                            req.getBatchSize(),
                            req.getMinWorkers(),
                            req.getMaxWorkers());
            workflow = new Workflow(modelInfo);
        }
        final ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<Void> f =
                modelManager
                        .registerWorkflow(workflow)
                        .exceptionally(
                                t -> {
                                    NettyUtils.sendError(ctx, t.getCause());
                                    if (req.isSynchronous()) {
                                        String name = workflow.getName();
                                        modelManager.unregisterWorkflow(name, req.getVersion());
                                    }
                                    return null;
                                });
        if (req.isSynchronous()) {
            final String msg = "Model \"" + req.getModelName() + "\" registered.";
            f.thenAccept(v -> NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg)));
        } else {
            String msg = "Model \"" + req.getModelName() + "\" registration scheduled.";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg), HttpResponseStatus.ACCEPTED);
        }
    }

    private void handleRegisterWorkflow(
            final ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String workflowUrl = NettyUtils.getParameter(decoder, LoadModelRequest.URL, null);
        if (workflowUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        boolean synchronous =
                Boolean.parseBoolean(
                        NettyUtils.getParameter(decoder, LoadModelRequest.SYNCHRONOUS, "true"));

        try {
            URI uri = URI.create(workflowUrl);
            Workflow workflow = WorkflowDefinition.parse(null, uri).toWorkflow();
            String workflowName = workflow.getName();

            final ModelManager modelManager = ModelManager.getInstance();
            CompletableFuture<Void> f =
                    modelManager
                            .registerWorkflow(workflow)
                            .exceptionally(
                                    t -> {
                                        NettyUtils.sendError(ctx, t.getCause());
                                        return null;
                                    });

            if (synchronous) {
                final String msg = "Workflow \"" + workflowName + "\" registered.";
                f.thenAccept(m -> NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg)));
            } else {
                String msg = "Workflow \"" + workflowName + "\" registration scheduled.";
                NettyUtils.sendJsonResponse(
                        ctx, new StatusResponse(msg), HttpResponseStatus.ACCEPTED);
            }
        } catch (IOException | BadWorkflowException e) {
            NettyUtils.sendError(ctx, e.getCause());
        }
    }

    private void handleUnregisterWorkflow(
            ChannelHandlerContext ctx, String workflowName, String version)
            throws ModelNotFoundException {
        ModelManager modelManager = ModelManager.getInstance();
        if (!modelManager.unregisterWorkflow(workflowName, version)) {
            ModelNotFoundException t =
                    new ModelNotFoundException("Model or workflow not found: " + workflowName);
            NettyUtils.sendError(ctx, t);
            throw t;
        }
        String msg = "Model or workflow \"" + workflowName + "\" unregistered";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void handleScaleWorkflow(
            ChannelHandlerContext ctx,
            QueryStringDecoder decoder,
            String workflowName,
            String version)
            throws ModelNotFoundException {
        try {
            String deviceName = NettyUtils.getParameter(decoder, LoadModelRequest.DEVICE, null);
            int minWorkers = NettyUtils.getIntParameter(decoder, LoadModelRequest.MIN_WORKER, -1);
            int maxWorkers = NettyUtils.getIntParameter(decoder, LoadModelRequest.MAX_WORKER, -1);

            ModelManager modelManager = ModelManager.getInstance();
            Endpoint endpoint = modelManager.getEndpoints().get(workflowName);
            List<Workflow> workflows = null;
            if (endpoint != null) {
                if (version == null) {
                    // scale all versions
                    workflows = endpoint.getWorkflows();
                } else {
                    Workflow wf = modelManager.getWorkflow(workflowName, version, false);
                    if (wf != null) {
                        workflows = Collections.singletonList(wf);
                    }
                }
            }
            if (workflows == null || workflows.isEmpty()) {
                throw new ModelNotFoundException("Model or workflow not found: " + workflowName);
            }

            List<String> messages = new ArrayList<>();
            for (Workflow workflow : workflows) {
                // make sure all WorkerPoolConfigs (models) are loaded and ready
                for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
                    if (wpc.getStatus() != WorkerPoolConfig.Status.READY) {
                        throw new ServiceUnavailableException(
                                "Model or workflow is not ready: " + workflow.getName());
                    }
                }

                for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
                    modelManager.scaleWorkers(wpc, deviceName, minWorkers, maxWorkers);
                    String msg =
                            "Workflow \""
                                    + workflow.getName()
                                    + "\" worker scaled. New Worker configuration min workers:"
                                    + minWorkers
                                    + " max workers:"
                                    + maxWorkers;
                    messages.add(msg);
                }
            }

            String combinedMsg = String.join("\n", messages);
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(combinedMsg));
        } catch (NumberFormatException ex) {
            throw new BadRequestException("parameter is invalid number." + ex.getMessage(), ex);
        }
    }

    private static final class ListPagination {

        private int pageToken;
        private int last;

        private ListPagination(QueryStringDecoder decoder, int keysSize) {
            int limit = NettyUtils.getIntParameter(decoder, "limit", 100);
            pageToken = NettyUtils.getIntParameter(decoder, "next_page_token", 0);
            if (limit > 100 || limit < 0) {
                limit = 100;
            }
            if (pageToken < 0) {
                pageToken = 0;
            }

            last = pageToken + limit;
            if (last > keysSize) {
                last = keysSize;
            }
        }
    }
}
