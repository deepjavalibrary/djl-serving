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
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeModelResponse;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerThread;
import ai.djl.serving.wlm.util.WlmCapacityException;
import ai.djl.serving.wlm.util.WlmShutdownException;
import ai.djl.translate.TranslateException;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
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
    public CompletableFuture<ModelInfo> registerModel(
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
                        ModelInfo modelInfo =
                                new ModelInfo(
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
                        if (!endpoint.add(modelInfo)) {
                            // model already exists
                            model.close();
                            throw new BadRequestException(
                                    "Model " + modelInfo + " is already registered.");
                        }

                        return modelInfo;
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
            for (ModelInfo m : endpoint.getModels()) {
                m.scaleWorkers(0, 0);
                wlm.modelChanged(m);
                startupModels.remove(modelName);
                m.close();
            }
            endpoint.getModels().clear();
            logger.info("Model {} unregistered.", modelName);
        } else {
            ModelInfo model = endpoint.remove(version);
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
     * trigger that a ModelInfo has been updated. Updates model workers for this model and scales
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
    public ModelInfo getModel(String modelName, String version, boolean predict) {
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
     * @param ctx the netty channel handler context where the job response will be sent
     * @param job an inference job to be executed
     */
    public void runJob(ChannelHandlerContext ctx, Job job) {
        wlm.runJob(job)
                .whenComplete(
                        (o, t) -> {
                            if (t != null) {
                                onException(t, ctx);
                            } else {
                                sendOutput(o, ctx);
                            }
                            logger.trace(
                                    "Waiting time: {}, Backend time: {}",
                                    job.getScheduled() - job.getBegin(),
                                    System.currentTimeMillis() - job.getScheduled());
                        })
                .exceptionally(
                        t -> {
                            onException(t, ctx);
                            return null;
                        });
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
        ModelInfo model = getModel(modelName, version, false);
        if (model == null) {
            throw new ModelNotFoundException("Model not found: " + modelName);
        }

        DescribeModelResponse resp = new DescribeModelResponse();
        resp.setModelName(modelName);
        resp.setModelUrl(model.getModelUrl());
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
                        for (ModelInfo m : endpoint.getModels()) {
                            numScaled += m.getMinWorkers();
                            numWorking += wlm.getNumRunningWorkers(m);
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

    void sendOutput(Output output, ChannelHandlerContext ctx) {
        FullHttpResponse resp =
                new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, HttpResponseStatus.OK, false);
        for (Map.Entry<String, String> entry : output.getProperties().entrySet()) {
            resp.headers().set(entry.getKey(), entry.getValue());
        }
        BytesSupplier data = output.getData();
        if (data != null) {
            resp.content().writeBytes(data.getAsBytes());
        }

        /*
         * We can load the models based on the configuration file.Since this Job is
         * not driven by the external connections, we could have a empty context for
         * this job. We shouldn't try to send a response to ctx if this is not triggered
         * by external clients.
         */
        if (ctx != null) {
            NettyUtils.sendHttpResponse(ctx, resp, true);
        }
    }

    void onException(Throwable t, ChannelHandlerContext ctx) {
        HttpResponseStatus status;
        if (t instanceof TranslateException) {
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmShutdownException) {
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
            logger.error("Unable to process prediction. Worker shutdown");
        } else if (t instanceof WlmCapacityException) {
            logger.error("Unable to process prediction. Worker capacity exceeded");
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            logger.warn("Unexpected error", t);
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
