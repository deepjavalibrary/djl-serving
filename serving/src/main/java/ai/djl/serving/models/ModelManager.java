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
import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeModelResponse;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkLoadManager.WorkerPool;
import ai.djl.serving.wlm.WorkerThread;
import ai.djl.serving.workflow.Workflow;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A class that in charge of managing models. */
public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);

    private static ModelManager modelManager;

    private ConfigManager configManager;
    private WorkLoadManager wlm;
    private Map<String, Endpoint> endpoints;
    private Set<String> startupModels;

    private ModelManager(ConfigManager configManager) {
        this.configManager = configManager;
        wlm = new WorkLoadManager();
        endpoints = new ConcurrentHashMap<>();
        startupModels = new HashSet<>();
    }

    /**
     * Initialized the global {@code ModelManager} instance.
     *
     * @param configManager the configuration
     */
    public static void init(ConfigManager configManager) {
        modelManager = new ModelManager(configManager);
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
     * Registers and loads a model.
     *
     * @param modelName the name of the model for HTTP endpoint
     * @param version the model version
     * @param modelUrl the model url
     * @param engineName the engine to load the model
     * @param deviceName the accelerator device id, -1 for auto selection
     * @param batchSize the batch size
     * @param maxBatchDelay the maximum delay for batching
     * @param maxIdleTime the maximum idle time of the worker threads before scaling down.
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<WorkflowInfo> registerWorkflow(
            final String modelName,
            final String version,
            final String modelUrl,
            final String engineName,
            final String deviceName,
            final int batchSize,
            final int maxBatchDelay,
            final int maxIdleTime) {
        return CompletableFuture.supplyAsync(
                        () -> {
                            try {
                                if (engineName != null) {
                                    DependencyManager dm = DependencyManager.getInstance();
                                    dm.installEngine(engineName);
                                }
                                Criteria.Builder<Input, Output> builder =
                                        Criteria.builder()
                                                .setTypes(Input.class, Output.class)
                                                .optModelUrls(modelUrl)
                                                .optEngine(engineName);
                                if ("-1".equals(deviceName)) {
                                    logger.info("Loading model {} on {}.", modelName, Device.cpu());
                                } else if (deviceName.startsWith("nc")) {
                                    logger.info("Loading model {} on {}.", modelName, deviceName);
                                    String ncs = deviceName.substring(2);
                                    builder.optOption("env", "NEURON_RT_VISIBLE_CORES=" + ncs);
                                } else {
                                    // GPU case
                                    int gpuId = Integer.parseInt(deviceName);
                                    builder.optDevice(Device.gpu(gpuId));
                                    logger.info(
                                            "Loading model {} on {}.",
                                            modelName,
                                            Device.gpu(gpuId));
                                }
                                if (batchSize > 1) {
                                    builder.optArgument("batchifier", "stack");
                                }

                                ZooModel<Input, Output> model = builder.build().loadModel();
                                return new WorkflowInfo(
                                        modelName,
                                        version,
                                        modelUrl,
                                        new ModelInfo(
                                                modelName,
                                                version,
                                                model,
                                                configManager.getJobQueueSize(),
                                                maxIdleTime,
                                                maxBatchDelay,
                                                batchSize));
                            } catch (ModelException | IOException e) {
                                throw new CompletionException(e);
                            }
                        })
                .thenApply(p -> registerWorkflow(p).join());
    }

    /**
     * Registers and loads a workflow.
     *
     * @param workflow the workflow to register
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<WorkflowInfo> registerWorkflow(final WorkflowInfo workflow) {
        return CompletableFuture.supplyAsync(
                () -> {
                    Endpoint endpoint =
                            endpoints.computeIfAbsent(workflow.getName(), k -> new Endpoint());
                    if (!endpoint.add(workflow)) {
                        // workflow already exists
                        throw new BadRequestException(
                                "Workflow " + workflow + " is already registered.");
                    }

                    return workflow;
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
        Set<ModelInfo> candidateModelsToUnregister = new HashSet<>();
        if (version == null) {
            // unregister all versions
            for (WorkflowInfo workflow : endpoint.getWorkflows()) {
                candidateModelsToUnregister.addAll(workflow.getWorkflow().getModels());
                workflow.getWorkflow().close();
            }
            startupModels.remove(workflowName);
            endpoint.getWorkflows().clear();
            logger.info("Model {} unregistered.", workflowName);
        } else {
            WorkflowInfo workflow = endpoint.remove(version);
            if (workflow == null) {
                logger.warn("Workflow not found: " + workflowName + ':' + version);
                return false;
            }
            candidateModelsToUnregister.addAll(workflow.getWorkflow().getModels());
            workflow.getWorkflow().close();
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
    public WorkflowInfo scaleWorkers(
            WorkflowInfo workflow, String deviceName, int minWorkers, int maxWorkers) {
        for (ModelInfo model : workflow.getWorkflow().getModels()) {
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
        String modelName = model.getModelName();
        logger.debug("updateModel: {}", modelName);
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
                .flatMap(w -> w.getWorkflow().getModels().stream())
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
    public WorkflowInfo getWorkflow(String workflowName, String version, boolean predict) {
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
        List<WorkflowInfo> list = endpoint.getWorkflows();
        if (list.isEmpty()) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }

        List<DescribeModelResponse> resps = new ArrayList<>();
        for (WorkflowInfo workflow : list) {
            for (ModelInfo model : workflow.getWorkflow().getModels()) {
                DescribeModelResponse resp = new DescribeModelResponse();
                resp.setModelName(model.getModelName());
                resp.setModelUrl(list.get(0).getModelUrl());
                resp.setBatchSize(model.getBatchSize());
                resp.setMaxBatchDelay(model.getMaxBatchDelay());
                resp.setMaxIdleTime(model.getMaxIdleTime());
                resp.setQueueLength(wlm.getQueueLength(model));
                resp.setLoadedAtStartup(startupModels.contains(model.getModelName()));

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
                        for (WorkflowInfo p : endpoint.getWorkflows()) {
                            for (ModelInfo m : p.getWorkflow().getModels()) {
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
