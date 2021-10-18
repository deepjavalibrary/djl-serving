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
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerThread;
import java.io.IOException;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
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
     * @param gpuId the GPU device id, -1 for auto selection
     * @param batchSize the batch size
     * @param maxBatchDelay the maximum delay for batching
     * @param maxIdleTime the maximum idle time of the worker threads before scaling down.
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<ServingModel> registerModel(
            final String modelName,
            final String version,
            final String modelUrl,
            final String engineName,
            final int gpuId,
            final int batchSize,
            final int maxBatchDelay,
            final int maxIdleTime) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        Criteria.Builder<Input, Output> builder =
                                Criteria.builder()
                                        .setTypes(Input.class, Output.class)
                                        .optModelUrls(modelUrl)
                                        .optEngine(engineName);
                        if (gpuId != -1) {
                            builder.optDevice(Device.gpu(gpuId));
                            logger.info("Loading model {} on {}.", modelName, Device.gpu(gpuId));
                        } else {
                            logger.info("Loading model {} on {}.", modelName, Device.cpu());
                        }
                        if (batchSize > 1) {
                            builder.optArgument("batchifier", "stack");
                        }

                        ZooModel<Input, Output> model = builder.build().loadModel();
                        ServingModel sm =
                                new ServingModel(
                                        modelName,
                                        version,
                                        modelUrl,
                                        model,
                                        configManager.getJobQueueSize(),
                                        maxIdleTime,
                                        maxBatchDelay,
                                        batchSize);

                        Endpoint endpoint =
                                endpoints.computeIfAbsent(modelName, k -> new Endpoint());
                        if (!endpoint.add(sm)) {
                            // model already exists
                            model.close();
                            throw new BadRequestException(
                                    "Model " + sm + " is already registered.");
                        }

                        return sm;
                    } catch (ModelException | IOException e) {
                        throw new CompletionException(e);
                    }
                });
    }

    /**
     * Unregisters a model by its name and version.
     *
     * @param modelName the model name to be unregistered
     * @param version the model version
     * @return {@code true} if unregister success
     */
    public boolean unregisterModel(String modelName, String version) {
        Endpoint endpoint = endpoints.get(modelName);
        if (endpoint == null) {
            logger.warn("Model not found: " + modelName);
            return false;
        }
        if (version == null) {
            // unregister all versions
            for (ServingModel sm : endpoint.getModels()) {
                ModelInfo m = sm.getModelInfo();
                m.scaleWorkers(0, 0);
                wlm.modelChanged(m);
                startupModels.remove(modelName);
                m.close();
            }
            endpoint.getModels().clear();
            logger.info("Model {} unregistered.", modelName);
        } else {
            ModelInfo model = endpoint.remove(version).getModelInfo();
            if (model == null) {
                logger.warn("Model not found: " + modelName + ':' + version);
                return false;
            }
            model.scaleWorkers(0, 0);
            wlm.modelChanged(model);
            startupModels.remove(modelName);
            model.close();
        }
        if (endpoint.getModels().isEmpty()) {
            endpoints.remove(modelName);
        }
        return true;
    }

    /**
     * trigger that a model has been updated. Updates model workers for this model and scales
     * up/down all workers to match the parameters for the model.
     *
     * @param modelInfo the model that has been updated
     * @return the model
     */
    public ModelInfo triggerModelUpdated(ModelInfo modelInfo) {
        String modelName = modelInfo.getModelName();
        logger.debug("updateModel: {}", modelName);
        wlm.modelChanged(modelInfo);
        return modelInfo;
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
     * Returns a version of model.
     *
     * @param modelName the model name
     * @param version the model version
     * @param predict ture for selecting a model in load balance fashion
     * @return the model
     */
    public ServingModel getModel(String modelName, String version, boolean predict) {
        Endpoint endpoint = endpoints.get(modelName);
        if (endpoint == null) {
            return null;
        }
        if (version == null) {
            if (endpoint.getModels().isEmpty()) {
                return null;
            }
            if (predict) {
                return endpoint.next();
            }
            return endpoint.getModels().get(0);
        }
        return endpoint.get(version);
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
     * @param job an inference job to be executed
     * @return {@code true} if submit success, false otherwise.
     */
    public CompletableFuture<Output> runJob(Job job) {
        return wlm.runJob(job);
    }

    /**
     * Returns a list of worker information for specified model.
     *
     * @param modelName the model name to be queried
     * @param version the model version to be queried
     * @return a list of worker information for specified model
     * @throws ModelNotFoundException if specified model not found
     */
    public DescribeModelResponse describeModel(String modelName, String version)
            throws ModelNotFoundException {
        ServingModel sm = getModel(modelName, version, false);
        if (sm == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }
        ModelInfo model = sm.getModelInfo();
        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(sm.getModelUrl());
        resp.setBatchSize(model.getBatchSize());
        resp.setMaxBatchDelay(model.getMaxBatchDelay());
        resp.setMaxWorkers(model.getMaxWorkers());
        resp.setMinWorkers(model.getMinWorkers());
        resp.setMaxIdleTime(model.getMaxIdleTime());
        resp.setQueueLength(wlm.getQueueLength(model));
        resp.setLoadedAtStartup(startupModels.contains(modelName));

        int activeWorker = wlm.getNumRunningWorkers(model);
        int targetWorker = model.getMinWorkers();
        resp.setStatus(activeWorker >= targetWorker ? "Healthy" : "Unhealthy");

        List<WorkerThread> workers = wlm.getWorkers(model);
        for (WorkerThread worker : workers) {
            int workerId = worker.getWorkerId();
            long startTime = worker.getStartTime();
            boolean isRunning = worker.isRunning();
            int gpuId = worker.getGpuId();
            resp.addWorker(workerId, startTime, isRunning, gpuId);
        }
        return resp;
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
                        for (ServingModel m : endpoint.getModels()) {
                            numScaled += m.getModelInfo().getMinWorkers();
                            numWorking += wlm.getNumRunningWorkers(m.getModelInfo());
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
