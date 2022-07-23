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
package ai.djl.serving.models;

import ai.djl.Device;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeWorkflowResponse;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.MutableClassLoader;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkLoadManager.WorkerPool;
import ai.djl.serving.wlm.WorkerThread;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.JsonUtils;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

/** A class that in charge of managing models. */
public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager = new ModelManager();

    private WorkLoadManager wlm;
    private Map<String, Endpoint> endpoints;
    private Set<String> startupWorkflows;

    private ModelManager() {
        wlm = new WorkLoadManager();
        endpoints = new ConcurrentHashMap<>();
        startupWorkflows = new HashSet<>();
    }

    /**
     * Returns the singleton {@code ModelManager} instance.
     *
     * @return the singleton {@code ModelManager} instance
     */
    public static ModelManager getInstance() {
        return modelManager;
    }

    /**
     * Registers and loads a {@link Workflow}.
     *
     * @param workflow the workflow to register
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<Void> registerWorkflow(Workflow workflow) {
        Endpoint endpoint = endpoints.computeIfAbsent(workflow.getName(), k -> new Endpoint());
        if (!endpoint.add(workflow)) {
            // workflow already exists
            throw new BadRequestException("Workflow " + workflow + " is already registered.");
        }

        return CompletableFuture.supplyAsync(
                () -> {
                    for (ModelInfo<Input, Output> model : workflow.getModels()) {
                        try {
                            // Install engine if necessary
                            String engine = model.getEngineName();
                            if (engine != null) {
                                DependencyManager dm = DependencyManager.getInstance();
                                dm.installEngine(engine);
                            }
                            wlm.registerModel(model);
                        } catch (IOException e) {
                            throw new CompletionException(e);
                        }
                    }
                    return null;
                });
    }

    /**
     * Unregisters a workflow by its name and version.
     *
     * @param workflowName the workflow name to be unregistered (may also be the same as a model
     *     name)
     * @param version the model version
     * @return {@code true} if unregister success
     */
    public boolean unregisterWorkflow(String workflowName, String version) {
        Endpoint endpoint = endpoints.get(workflowName);
        if (endpoint == null) {
            logger.warn("Model not found: " + workflowName);
            return false;
        }
        Set<ModelInfo<Input, Output>> candidateModelsToUnregister = new HashSet<>();
        if (version == null) {
            // unregister all versions
            for (Workflow workflow : endpoint.getWorkflows()) {
                candidateModelsToUnregister.addAll(workflow.getModels());
                workflow.close();
            }
            startupWorkflows.remove(workflowName);
            endpoint.getWorkflows().clear();
            logger.info("Model {} unregistered.", workflowName);
        } else {
            Workflow workflow = endpoint.remove(version);
            if (workflow == null) {
                logger.warn("Workflow not found: " + workflowName + ':' + version);
                return false;
            }
            candidateModelsToUnregister.addAll(workflow.getModels());
            workflow.close();
            startupWorkflows.remove(workflowName);
        }
        if (endpoint.getWorkflows().isEmpty()) {
            endpoints.remove(workflowName);
        }

        // Unregister candidate models if they are not used for a remaining endpoint
        candidateModelsToUnregister.removeAll(getModels());
        for (ModelInfo<Input, Output> model : candidateModelsToUnregister) {
            wlm.unregisterModel(model);
        }

        return true;
    }

    /**
     * Scales the workers for each model in a workflow.
     *
     * @param workflow the workflow to scale workers for
     * @param device the device for the model
     * @param minWorkers the min workers
     * @param maxWorkers the max workers
     * @return the info about the scaled workflow
     * @see WorkerPool#scaleWorkers(Device, int, int)
     */
    public Workflow scaleWorkers(Workflow workflow, Device device, int minWorkers, int maxWorkers) {
        for (ModelInfo<Input, Output> model : workflow.getModels()) {
            scaleWorkers(model, device, minWorkers, maxWorkers);
        }
        return workflow;
    }

    /**
     * Scales the workers for a model.
     *
     * @param model the model to scale workers for
     * @param device the device for the model
     * @param minWorkers the min workers
     * @param maxWorkers the max workers
     * @return the info about the scaled workflow
     * @see WorkerPool#scaleWorkers(Device, int, int)
     */
    public ModelInfo<Input, Output> scaleWorkers(
            ModelInfo<Input, Output> model, Device device, int minWorkers, int maxWorkers) {
        logger.debug("updateModel: {}", model);
        Thread.currentThread().setContextClassLoader(MutableClassLoader.getInstance());
        wlm.getWorkerPoolForModel(model).scaleWorkers(device, minWorkers, maxWorkers);
        return model;
    }

    /**
     * Returns the registry of all endpoints.
     *
     * @return the registry of all endpoints
     */
    public Map<String, Endpoint> getEndpoints() {
        return endpoints;
    }

    /**
     * Returns all models in an endpoint.
     *
     * @return all models in an endpoint
     */
    public Set<ModelInfo<Input, Output>> getModels() {
        return getEndpoints().values().stream()
                .flatMap(e -> e.getWorkflows().stream())
                .flatMap(w -> w.getModels().stream())
                .collect(Collectors.toSet());
    }

    /**
     * Returns a version of workflow.
     *
     * @param workflowName the workflow name
     * @param version the model version
     * @param predict ture for selecting a model in load balance fashion
     * @return the model
     */
    public Workflow getWorkflow(String workflowName, String version, boolean predict) {
        Endpoint endpoint = endpoints.get(workflowName);
        if (endpoint == null) {
            return null;
        }
        if (version == null) {
            if (endpoint.getWorkflows().isEmpty()) {
                return null;
            }
            if (predict) {
                return endpoint.next();
            }
            return endpoint.getWorkflows().get(0);
        }
        return endpoint.get(version);
    }

    /**
     * Returns the {@link WorkLoadManager}.
     *
     * @return the {@link WorkLoadManager}
     */
    public WorkLoadManager getWorkLoadManager() {
        return wlm;
    }

    /**
     * Returns a set of models or workflows that were loaded at startup.
     *
     * @return a set of models or workflows that were loaded at startup
     */
    public Set<String> getStartupWorkflows() {
        return startupWorkflows;
    }

    /**
     * Runs an inference job by assigning the job to the next free worker.
     *
     * @param workflow the workflow to run
     * @param input the input to the task
     * @return {@code true} if submit success, false otherwise.
     */
    public CompletableFuture<Output> runJob(Workflow workflow, Input input) {
        return workflow.execute(wlm, input);
    }

    /**
     * Returns a list of worker information for specified workflow.
     *
     * @param workflowName the workflow name to be queried
     * @param version the model version to be queried
     * @return a list of worker information for specified workflow
     * @throws ModelNotFoundException if specified workflow not found
     */
    public List<DescribeWorkflowResponse> describeWorkflow(String workflowName, String version)
            throws ModelNotFoundException {
        Endpoint endpoint = endpoints.get(workflowName);
        if (endpoint == null) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }
        List<Workflow> list = endpoint.getWorkflows();
        if (list.isEmpty()) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }

        List<DescribeWorkflowResponse> resps = new ArrayList<>();
        for (Workflow workflow : list) {
            for (ModelInfo<Input, Output> model : workflow.getModels()) {
                DescribeWorkflowResponse resp = new DescribeWorkflowResponse();
                resp.setWorkflowName(model.getModelId());
                resp.setWorkflowUrl(model.getModelUrl());
                resp.setBatchSize(model.getBatchSize());
                resp.setMaxBatchDelay(model.getMaxBatchDelay());
                resp.setMaxIdleTime(model.getMaxIdleTime());
                resp.setQueueLength(wlm.getQueueLength(model));
                resp.setLoadedAtStartup(startupWorkflows.contains(model.getModelId()));

                WorkerPool<Input, Output> wp = wlm.getWorkerPoolForModel(model);
                resp.setMaxWorkers(wp.getMaxWorkers());
                resp.setMinWorkers(wp.getMinWorkers());

                ModelInfo.Status status = model.getStatus();
                if (status == ModelInfo.Status.READY) {
                    int activeWorker = wlm.getNumRunningWorkers(model);
                    int targetWorker = wp.getMinWorkers();
                    resp.setStatus(activeWorker >= targetWorker ? "Healthy" : "Unhealthy");
                } else {
                    resp.setStatus(status.name());
                }

                List<WorkerThread<Input, Output>> workers = wlm.getWorkers(model);
                for (WorkerThread<Input, Output> worker : workers) {
                    int workerId = worker.getWorkerId();
                    long startTime = worker.getStartTime();
                    boolean isRunning = worker.isRunning();
                    resp.addWorker(
                            model.getVersion(), workerId, startTime, isRunning, worker.getDevice());
                }
                resps.add(resp);
            }
        }

        return resps;
    }

    /**
     * Sends model server health status to client.
     *
     * @return completableFuture with eventually result in the future after async execution
     */
    public CompletableFuture<FullHttpResponse> workerStatus(String type) {
        return CompletableFuture.supplyAsync(
                () -> {
                    boolean hasFailure = false;
                    boolean hasPending = false;
                    Map<String, StatusResponse> data = new LinkedHashMap<>(); // NOPMD
                    for (Endpoint endpoint : endpoints.values()) {
                        for (Workflow p : endpoint.getWorkflows()) {
                            String workflowName = p.getName();
                            for (ModelInfo<Input, Output> m : p.getModels()) {
                                String modelName = m.getModelId();
                                if (!modelName.equals(workflowName)) {
                                    modelName = workflowName + ':' + modelName; // NOPMD
                                }
                                ModelInfo.Status status = m.getStatus();
                                switch (status) {
                                    case FAILED:
                                        data.put(modelName, new StatusResponse(status.name()));
                                        hasFailure = true;
                                        break;
                                    case PENDING:
                                        data.put(modelName, new StatusResponse(status.name()));
                                        hasPending = true;
                                        break;
                                    default:
                                        int min = wlm.getWorkerPoolForModel(m).getMinWorkers();
                                        int actual = wlm.getNumRunningWorkers(m);
                                        if (actual < min) {
                                            data.put(modelName, new StatusResponse("Unhealthy"));
                                        } else {
                                            data.put(modelName, new StatusResponse("Healthy"));
                                        }
                                        break;
                                }
                            }
                        }
                    }

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

                    FullHttpResponse resp =
                            new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
                    resp.headers()
                            .set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
                    ByteBuf content = resp.content();
                    String body = null;
                    if(type.equals("FOR_KSERVE")){
                        // empty response
                        body = JsonUtils.GSON_PRETTY.toJson(new LinkedHashMap<>());
                    }
                    else if(type.equals("FOR_DJL_SERVING")){
                        body = JsonUtils.GSON_PRETTY.toJson(data);
                    }
                    content.writeCharSequence(body, CharsetUtil.UTF_8);
                    content.writeByte('\n');
                    return resp;
                });
    }

    /**
     * Clears everything in the {@link ModelManager}.
     *
     * <p>Can be run between tests.
     */
    public void clear() {
        wlm.close();
        for (Endpoint endpoint : endpoints.values()) {
            endpoint.close();
        }

        wlm = new WorkLoadManager();
        endpoints = new ConcurrentHashMap<>();
        startupWorkflows = new HashSet<>();
    }
}
