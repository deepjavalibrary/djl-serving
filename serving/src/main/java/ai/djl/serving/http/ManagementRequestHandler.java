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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager.WorkerPool;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.apache.logging.log4j.util.Strings;

import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
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

    /** HTTP Parameter "synchronous". */
    private static final String SYNCHRONOUS_PARAMETER = "synchronous";
    /** HTTP Parameter "url". */
    private static final String URL_PARAMETER = "url";
    /** HTTP Parameter "batch_size". */
    private static final String BATCH_SIZE_PARAMETER = "batch_size";
    /** HTTP Parameter "model_name". */
    private static final String MODEL_NAME_PARAMETER = "model_name";
    /** HTTP Parameter "model_version". */
    private static final String MODEL_VERSION_PARAMETER = "model_version";
    /** HTTP Parameter "engine". */
    private static final String ENGINE_NAME_PARAMETER = "engine";
    /** HTTP Parameter "device". */
    private static final String DEVICE_PARAMETER = "device";
    /** HTTP Parameter "max_batch_delay". */
    private static final String MAX_BATCH_DELAY_PARAMETER = "max_batch_delay";
    /** HTTP Parameter "max_idle_time". */
    private static final String MAX_IDLE_TIME_PARAMETER = "max_idle_time";
    /** HTTP Parameter "max_worker". */
    private static final String MAX_WORKER_PARAMETER = "max_worker";
    /** HTTP Parameter "min_worker". */
    private static final String MIN_WORKER_PARAMETER = "min_worker";

    private static final Pattern WORKFLOWS_PATTERN = Pattern.compile("^/workflows([/?].*)?");
    private static final Pattern MODELS_PATTERN = Pattern.compile("^/models([/?].*)?");
    private static final Pattern KSERVEV2_PATTERN = Pattern.compile("^/v2([/?].*)?");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return WORKFLOWS_PATTERN.matcher(req.uri()).matches()
                    || MODELS_PATTERN.matcher(req.uri()).matches()
                    || KSERVEV2_PATTERN.matcher(req.uri()).matches();
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
                    handleRegisterModel(ctx, decoder);
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

    private boolean isKFV2InferenceReq(String[] segments) {
        return segments.length == 5
                && "v2".equals(segments[1])
                && "models".equals(segments[2])
                && (segments[4].equals("infer") || segments[4].equals("explain"));
    }

    private void handleListModels(ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        ModelManager modelManager = ModelManager.getInstance();
        Map<String, Endpoint> endpoints = modelManager.getEndpoints();

        List<String> keys = new ArrayList<>(endpoints.keySet());
        Collections.sort(keys);
        ListModelsResponse list = new ListModelsResponse();

        ListPagination pagination = new ListPagination(decoder, keys.size());
        if (pagination.last <= keys.size()) {
            list.setNextPageToken(String.valueOf(pagination.last));
        }

        for (int i = pagination.pageToken; i < pagination.last; ++i) {
            String workflowName = keys.get(i);
            for (Workflow workflow : endpoints.get(workflowName).getWorkflows()) {
                for (ModelInfo<Input, Output> m : workflow.getModels()) {
                    String status = m.getStatus().toString();
                    String id = m.getModelId();
                    String modelName;
                    if (workflowName.equals(id)) {
                        modelName = workflowName;
                    } else {
                        modelName = workflowName + ':' + id;
                    }
                    list.addModel(modelName, workflow.getVersion(), m.getModelUrl(), status);
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
        List<DescribeWorkflowResponse> resps = modelManager.describeWorkflow(workflowName, version);

        if (resps.size() != 1) {
            NettyUtils.sendError(
                    ctx,
                    new IllegalArgumentException(
                            workflowName + " describes a full workflow, not just a model"));
            return;
        }

        DescribeWorkflowResponse resp = resps.get(0);
        NettyUtils.sendJsonResponse(ctx, resp);
    }

    private void handleRegisterModel(final ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String modelUrl = NettyUtils.getParameter(decoder, URL_PARAMETER, null);
        if (modelUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String modelName = NettyUtils.getParameter(decoder, MODEL_NAME_PARAMETER, null);
        if (modelName == null || modelName.isEmpty()) {
            modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
        }
        String version = NettyUtils.getParameter(decoder, MODEL_VERSION_PARAMETER, null);
        String deviceName = NettyUtils.getParameter(decoder, DEVICE_PARAMETER, "-1");
        String engineName = NettyUtils.getParameter(decoder, ENGINE_NAME_PARAMETER, null);
        int batchSize = NettyUtils.getIntParameter(decoder, BATCH_SIZE_PARAMETER, 1);
        int maxBatchDelay = NettyUtils.getIntParameter(decoder, MAX_BATCH_DELAY_PARAMETER, 100);
        int maxIdleTime = NettyUtils.getIntParameter(decoder, MAX_IDLE_TIME_PARAMETER, 60);
        int minWorkers = NettyUtils.getIntParameter(decoder, MIN_WORKER_PARAMETER, 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, MAX_WORKER_PARAMETER, -1);
        boolean synchronous =
                Boolean.parseBoolean(
                        NettyUtils.getParameter(decoder, SYNCHRONOUS_PARAMETER, "true"));

        Engine engine = engineName != null ? Engine.getEngine(engineName) : Engine.getInstance();
        Device device = Device.fromName(deviceName, engine);
        ModelInfo<Input, Output> modelInfo =
                new ModelInfo<>(
                        modelName,
                        modelUrl,
                        version,
                        engineName,
                        Input.class,
                        Output.class,
                        WlmConfigManager.getInstance().getJobQueueSize(),
                        maxIdleTime,
                        maxBatchDelay,
                        batchSize);
        Workflow workflow = new Workflow(modelInfo);
        final ModelManager modelManager = ModelManager.getInstance();
        CompletableFuture<Void> f =
                modelManager
                        .registerWorkflow(workflow)
                        .thenAccept(
                                v -> {
                                    for (ModelInfo<Input, Output> m : workflow.getModels()) {
                                        m.configurePool(maxIdleTime)
                                                .configureModelBatch(batchSize, maxBatchDelay);
                                        modelManager.scaleWorkers(
                                                m, device, minWorkers, maxWorkers);
                                    }
                                })
                        .exceptionally(
                                t -> {
                                    NettyUtils.sendError(ctx, t.getCause());
                                    return null;
                                });
        if (synchronous) {
            final String msg = "Model \"" + modelName + "\" registered.";
            f.thenAccept(v -> NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg)));
        } else {
            String msg = "Model \"" + modelName + "\" registration scheduled.";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg), HttpResponseStatus.ACCEPTED);
        }
    }

    private void handleRegisterWorkflow(
            final ChannelHandlerContext ctx, QueryStringDecoder decoder) {
        String workflowUrl = NettyUtils.getParameter(decoder, URL_PARAMETER, null);
        if (workflowUrl == null) {
            throw new BadRequestException("Parameter url is required.");
        }

        String deviceName = NettyUtils.getParameter(decoder, DEVICE_PARAMETER, "-1");
        int minWorkers = NettyUtils.getIntParameter(decoder, MIN_WORKER_PARAMETER, 1);
        int maxWorkers = NettyUtils.getIntParameter(decoder, MAX_WORKER_PARAMETER, -1);
        boolean synchronous =
                Boolean.parseBoolean(
                        NettyUtils.getParameter(decoder, SYNCHRONOUS_PARAMETER, "true"));

        try {
            URL url = new URL(workflowUrl);
            Workflow workflow =
                    WorkflowDefinition.parse(url.toURI(), url.openStream()).toWorkflow();
            String workflowName = workflow.getName();

            Device device = Device.fromName(deviceName);
            final ModelManager modelManager = ModelManager.getInstance();
            CompletableFuture<Void> f =
                    modelManager
                            .registerWorkflow(workflow)
                            .thenAccept(
                                    v ->
                                            modelManager.scaleWorkers(
                                                    workflow, device, minWorkers, maxWorkers))
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

        } catch (URISyntaxException | IOException | BadWorkflowException e) {
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
            ModelManager modelManager = ModelManager.getInstance();
            Workflow workflow = modelManager.getWorkflow(workflowName, version, false);
            if (workflow == null) {
                throw new ModelNotFoundException("Model or workflow not found: " + workflowName);
            }

            // make sure all models are loaded and ready
            for (ModelInfo<Input, Output> modelInfo : workflow.getModels()) {
                if (modelInfo.getStatus() != ModelInfo.Status.READY) {
                    throw new ServiceUnavailableException(
                            "Model or workflow is not ready: " + workflowName);
                }
            }

            List<String> msgs = new ArrayList<>();
            for (ModelInfo<Input, Output> modelInfo : workflow.getModels()) {
                WorkerPool<Input, Output> pool =
                        modelManager.getWorkLoadManager().getWorkerPoolForModel(modelInfo);
                int minWorkers =
                        NettyUtils.getIntParameter(
                                decoder, MIN_WORKER_PARAMETER, pool.getMinWorkers());
                int maxWorkers =
                        NettyUtils.getIntParameter(
                                decoder, MAX_WORKER_PARAMETER, pool.getMaxWorkers());
                if (maxWorkers < minWorkers) {
                    throw new BadRequestException("max_worker cannot be less than min_worker.");
                }

                int maxIdleTime =
                        NettyUtils.getIntParameter(
                                decoder, MAX_IDLE_TIME_PARAMETER, modelInfo.getMaxIdleTime());
                int batchSize =
                        NettyUtils.getIntParameter(
                                decoder, BATCH_SIZE_PARAMETER, modelInfo.getBatchSize());
                int maxBatchDelay =
                        NettyUtils.getIntParameter(
                                decoder, MAX_BATCH_DELAY_PARAMETER, modelInfo.getMaxBatchDelay());

                if (version == null) {
                    // scale all versions
                    Endpoint endpoint = modelManager.getEndpoints().get(workflowName);
                    for (Workflow p : endpoint.getWorkflows()) {
                        for (ModelInfo<Input, Output> m : p.getModels()) {
                            m.configurePool(maxIdleTime)
                                    .configureModelBatch(batchSize, maxBatchDelay);
                            modelManager.scaleWorkers(m, null, minWorkers, maxWorkers);
                        }
                    }
                } else {
                    modelInfo
                            .configurePool(maxIdleTime)
                            .configureModelBatch(batchSize, maxBatchDelay);
                    modelManager.scaleWorkers(modelInfo, null, minWorkers, maxWorkers);
                }

                String msg =
                        "Workflow \""
                                + workflowName
                                + "\" worker scaled. New Worker configuration min workers:"
                                + pool.getMinWorkers()
                                + " max workers:"
                                + pool.getMaxWorkers();
                msgs.add(msg);
            }
            String combinedMsg = Strings.join(msgs, '\n');
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
