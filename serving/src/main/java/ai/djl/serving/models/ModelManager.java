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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeModelResponse;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkLoadManager.WorkerPool;
import ai.djl.serving.wlm.WorkerThread;
import ai.djl.serving.workflow.Workflow;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class that in charge of managing models. */
public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager = new ModelManager();

    private WorkLoadManager wlm;
    private Map<String, Endpoint> endpoints;
    private Set<String> startupModels;

    private ModelManager() {
        wlm = new WorkLoadManager();
        endpoints = new ConcurrentHashMap<>();
        startupModels = new HashSet<>();
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
     * @param deviceName the accelerator device id, -1 for auto selection
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<Void> registerWorkflow(Workflow workflow, String deviceName) {
        Endpoint endpoint = endpoints.computeIfAbsent(workflow.getName(), k -> new Endpoint());
        if (!endpoint.add(workflow)) {
            // workflow already exists
            throw new BadRequestException("Workflow " + workflow + " is already registered.");
        }
        return workflow.load(deviceName);
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
        Set<ModelInfo> candidateModelsToUnregister = new HashSet<>();
        if (version == null) {
            // unregister all versions
            for (Workflow workflow : endpoint.getWorkflows()) {
                candidateModelsToUnregister.addAll(workflow.getModels());
                workflow.close();
            }
            startupModels.remove(workflowName);
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
            startupModels.remove(workflowName);
        }
        if (endpoint.getWorkflows().isEmpty()) {
            endpoints.remove(workflowName);
        }

        // Unregister candidate models if they are not used for a remaining endpoint
        candidateModelsToUnregister.removeAll(getModels());
        for (ModelInfo model : candidateModelsToUnregister) {
            wlm.unregisterModel(model);
        }

        return true;
    }

    /**
     * Scales the workers for each model in a workflow.
     *
     * @param workflow the workflow to scale workers for
     * @param deviceName the device for the model
     * @param minWorkers the min workers
     * @param maxWorkers the max workers
     * @return the info about the scaled workflow
     * @see WorkerPool#scaleWorkers(String, int, int)
     */
    public Workflow scaleWorkers(
            Workflow workflow, String deviceName, int minWorkers, int maxWorkers) {
        for (ModelInfo model : workflow.getModels()) {
            scaleWorkers(model, deviceName, minWorkers, maxWorkers);
        }
        return workflow;
    }

    /**
     * Scales the workers for a model.
     *
     * @param model the model to scale workers for
     * @param deviceName the device for the model
     * @param minWorkers the min workers
     * @param maxWorkers the max workers
     * @return the info about the scaled workflow
     * @see WorkerPool#scaleWorkers(String, int, int)
     */
    public ModelInfo scaleWorkers(
            ModelInfo model, String deviceName, int minWorkers, int maxWorkers) {
        logger.debug("updateModel: {}", model);
        wlm.getWorkerPoolForModel(model).scaleWorkers(deviceName, minWorkers, maxWorkers);
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
    public Set<ModelInfo> getModels() {
        return getEndpoints()
                .values()
                .stream()
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
     * Returns a set of models that was loaded at startup.
     *
     * @return a set of models that was loaded at startup
     */
    public Set<String> getStartupModels() {
        return startupModels;
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
    public List<DescribeModelResponse> describeWorkflow(String workflowName, String version)
            throws ModelNotFoundException {
        Endpoint endpoint = endpoints.get(workflowName);
        if (endpoint == null) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }
        List<Workflow> list = endpoint.getWorkflows();
        if (list.isEmpty()) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }

        List<DescribeModelResponse> resps = new ArrayList<>();
        for (Workflow workflow : list) {
            for (ModelInfo model : workflow.getModels()) {
                DescribeModelResponse resp = new DescribeModelResponse();
                resp.setModelName(model.getModelId());
                resp.setModelUrl(model.getModelUrl());
                resp.setBatchSize(model.getBatchSize());
                resp.setMaxBatchDelay(model.getMaxBatchDelay());
                resp.setMaxIdleTime(model.getMaxIdleTime());
                resp.setQueueLength(wlm.getQueueLength(model));
                resp.setLoadedAtStartup(startupModels.contains(model.getModelId()));

                WorkerPool wp = wlm.getWorkerPoolForModel(model);
                resp.setMaxWorkers(wp.getMaxWorkers());
                resp.setMinWorkers(wp.getMinWorkers());

                int activeWorker = wlm.getNumRunningWorkers(model);
                int targetWorker = wp.getMinWorkers();
                resp.setStatus(activeWorker >= targetWorker ? "Healthy" : "Unhealthy");

                List<WorkerThread> workers = wlm.getWorkers(model);
                for (WorkerThread worker : workers) {
                    int workerId = worker.getWorkerId();
                    long startTime = worker.getStartTime();
                    boolean isRunning = worker.isRunning();
                    int gpuId = worker.getGpuId();
                    resp.addWorker(model.getVersion(), workerId, startTime, isRunning, gpuId);
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
    public CompletableFuture<String> workerStatus() {
        return CompletableFuture.supplyAsync(
                () -> {
                    String response = "Healthy";
                    int numWorking = 0;

                    int numScaled = 0;
                    for (Endpoint endpoint : endpoints.values()) {
                        for (Workflow p : endpoint.getWorkflows()) {
                            for (ModelInfo m : p.getModels()) {
                                numScaled += wlm.getWorkerPoolForModel(m).getMinWorkers();
                                numWorking += wlm.getNumRunningWorkers(m);
                            }
                        }
                    }

                    if ((numWorking > 0) && (numWorking < numScaled)) {
                        response = "Partial Healthy";
                    } else if ((numWorking == 0) && (numScaled > 0)) {
                        response = "Unhealthy";
                    }

                    return response;
                });
    }
}
