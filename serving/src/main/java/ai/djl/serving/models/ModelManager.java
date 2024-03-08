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

import ai.djl.ModelException;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.DescribeWorkflowResponse;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.util.MutableClassLoader;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPool;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.Workflow;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;

/** A class that in charge of managing models. */
public final class ModelManager {

    private static final Logger logger = LoggerFactory.getLogger(ModelManager.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

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
            throw new BadRequestException(409, "Workflow " + workflow + " is already registered.");
        }

        return CompletableFuture.supplyAsync(
                () -> {
                    long begin = System.nanoTime();
                    Map<String, WorkerPoolConfig<Input, Output>> wpcs = workflow.getWpcMap();
                    for (Map.Entry<String, WorkerPoolConfig<Input, Output>> entry :
                            wpcs.entrySet()) {
                        String key = entry.getKey();
                        WorkerPoolConfig<Input, Output> workerPoolConfig = entry.getValue();
                        try {
                            // download model and configure per model settings
                            workerPoolConfig.initialize();

                            // Install engine if necessary
                            String engine = null;
                            if (workerPoolConfig instanceof ModelInfo) {
                                ModelInfo<Input, Output> model =
                                        (ModelInfo<Input, Output>) workerPoolConfig;
                                engine = model.getEngineName();
                                DependencyManager dm = DependencyManager.getInstance();
                                dm.installEngine(engine);
                                Thread.currentThread()
                                        .setContextClassLoader(MutableClassLoader.getInstance());
                                WorkerPool<Input, Output> wp = wlm.getWorkerPool(model);
                                if (wp != null) {
                                    wpcs.put(key, wp.getWpc());
                                    wp.increaseRef();
                                    logger.info("Model {} is registered by other workflow", model);
                                    continue;
                                }
                            }

                            wlm.registerWorkerPool(workerPoolConfig);
                            String[] devices = workerPoolConfig.getLoadOnDevices();
                            if (engine != null) {
                                logger.info(
                                        "Loading model on {}:{}", engine, Arrays.toString(devices));
                            } else {
                                logger.info("Loading worker: {}", Arrays.toString(devices));
                            }
                            ExecutorService pool = null;
                            List<Future<?>> futures = new ArrayList<>();
                            if (workerPoolConfig.isParallelLoading()) {
                                pool = Executors.newFixedThreadPool(devices.length);
                            }
                            for (String deviceName : devices) {
                                if (pool != null) {
                                    futures.add(
                                            pool.submit(
                                                    () ->
                                                            initWorkers(
                                                                    workerPoolConfig, deviceName)));
                                } else {
                                    initWorkers(workerPoolConfig, deviceName);
                                }
                            }
                            if (pool != null) {
                                pool.shutdown();
                                for (Future<?> future : futures) {
                                    try {
                                        future.get();
                                    } catch (ExecutionException e) {
                                        throw new CompletionException(e.getCause()); // NOPMD
                                    } catch (InterruptedException e) {
                                        throw new AssertionError("Worker startup interrupted.", e);
                                    }
                                }
                            }
                        } catch (IOException | ModelException e) {
                            throw new CompletionException(e);
                        }
                    }
                    workflow.prepare(wlm);
                    long duration = (System.nanoTime() - begin) / 1000;
                    Dimension dimension = new Dimension("Model", workflow.getName());
                    Metric metric =
                            new Metric("RegisterWorkflow", duration, Unit.MICROSECONDS, dimension);
                    MODEL_METRIC.info("{}", metric);
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
            logger.warn("Model not found: {}", workflowName);
            return false;
        }
        Set<WorkerPoolConfig<Input, Output>> candidateWpcsToUnregister = new HashSet<>();
        if (version == null) {
            // unregister all versions
            for (Workflow workflow : endpoint.getWorkflows()) {
                candidateWpcsToUnregister.addAll(workflow.getWpcs());
                workflow.close();
            }
            startupWorkflows.remove(workflowName);
            endpoint.getWorkflows().clear();
            logger.info("Model {} unregistered.", workflowName);
        } else {
            Workflow workflow = endpoint.remove(version);
            if (workflow == null) {
                logger.warn("Workflow not found: {}:{}", workflowName, version);
                return false;
            }
            candidateWpcsToUnregister.addAll(workflow.getWpcs());
            workflow.close();
            startupWorkflows.remove(workflowName);
            logger.info("Model {}/{} unregistered.", workflowName, version);
        }
        if (endpoint.getWorkflows().isEmpty()) {
            endpoints.remove(workflowName);
        }

        // Unregister candidate models if they are not used for a remaining endpoint
        candidateWpcsToUnregister.removeAll(getWpcs());
        for (WorkerPoolConfig<Input, Output> wpc : candidateWpcsToUnregister) {
            wlm.unregisterWorkerPool(wpc);
        }

        return true;
    }

    /**
     * Initializes the workers for a workerPoolConfig.
     *
     * @param wpc the workerPoolConfig to scale workers for
     * @param deviceName the device for the workerPoolConfig
     * @see WorkerPool#initWorkers(String)
     */
    public void initWorkers(WorkerPoolConfig<Input, Output> wpc, String deviceName) {
        Thread.currentThread().setContextClassLoader(MutableClassLoader.getInstance());
        wlm.getWorkerPool(wpc).initWorkers(deviceName);
    }

    /**
     * Scales the workers for a model.
     *
     * @param wpc the model to scale workers for
     * @param deviceName the device for the model
     * @param minWorkers the min workers, -1 for auto-scale
     * @param maxWorkers the max workers, -1 for auto-scale
     * @see WorkerPool#scaleWorkers(String, int, int)
     */
    public void scaleWorkers(
            WorkerPoolConfig<Input, Output> wpc,
            String deviceName,
            int minWorkers,
            int maxWorkers) {
        Thread.currentThread().setContextClassLoader(MutableClassLoader.getInstance());
        wlm.getWorkerPool(wpc).scaleWorkers(deviceName, minWorkers, maxWorkers);
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
     * Returns all {@link WorkerPoolConfig}s in an endpoint.
     *
     * @return all {@link WorkerPoolConfig}s in an endpoint
     */
    public Set<WorkerPoolConfig<Input, Output>> getWpcs() {
        return getEndpoints().values().stream()
                .flatMap(e -> e.getWorkflows().stream())
                .flatMap(w -> w.getWpcs().stream())
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
     * Returns the single startup workflow.
     *
     * <p>Returns only if there was exactly 1 startup workflow passed in. Used with integration of
     * SageMaker SME and single model services.
     *
     * @return the workflow name
     */
    public Optional<String> getSingleStartupWorkflow() {
        Set<String> startModels = getStartupWorkflows();
        if (startModels.size() == 1) {
            return Optional.ofNullable(startModels.iterator().next());
        }
        return Optional.empty();
    }

    /**
     * Runs an inference job by assigning the job to the next free worker.
     *
     * @param workflow the workflow to run
     * @param input the input to the task
     * @return the {@code CompletableFuture}
     */
    public CompletableFuture<Output> runJob(Workflow workflow, Input input) {
        return workflow.execute(wlm, input);
    }

    /**
     * Returns a list of worker information for specified workflow.
     *
     * @param workflowName the workflow name to be queried
     * @param version the model version to be queried
     * @return model and workers information for specified workflow
     * @throws ModelNotFoundException if specified workflow not found
     */
    public DescribeWorkflowResponse[] describeWorkflow(String workflowName, String version)
            throws ModelNotFoundException {
        Endpoint endpoint = endpoints.get(workflowName);
        if (endpoint == null) {
            throw new ModelNotFoundException("Workflow not found: " + workflowName);
        }
        List<Workflow> list = null;
        if (version == null) {
            list = endpoint.getWorkflows();
        } else {
            Workflow wf = endpoint.get(version);
            if (wf != null) {
                list = Collections.singletonList(wf);
            }
        }
        if (list == null || list.isEmpty()) {
            StringBuilder sb = new StringBuilder("Workflow not found: ");
            sb.append(workflowName);
            if (version != null) {
                sb.append('/').append(version);
            }
            throw new ModelNotFoundException("Workflow not found: " + sb);
        }

        DescribeWorkflowResponse[] array = new DescribeWorkflowResponse[list.size()];
        int index = 0;
        for (Workflow workflow : list) {
            array[index++] = new DescribeWorkflowResponse(workflow);
        }

        return array;
    }

    /**
     * Sends model server health status to client.
     *
     * @return completableFuture with eventually result in the future after async execution
     */
    public CompletableFuture<Map<String, Object>> workerStatus() {
        return CompletableFuture.supplyAsync(
                () -> {
                    boolean hasFailure = false;
                    boolean hasPending = false;
                    Map<String, StatusResponse> data = new LinkedHashMap<>(); // NOPMD
                    for (Endpoint endpoint : endpoints.values()) {
                        for (Workflow wf : endpoint.getWorkflows()) {
                            String workflowName = wf.getName();
                            for (WorkerPoolConfig<Input, Output> wpc : wf.getWpcs()) {
                                String modelName = wpc.getId();
                                if (!modelName.equals(workflowName)) {
                                    modelName = workflowName + ':' + modelName; // NOPMD
                                }
                                WorkerPoolConfig.Status status = wpc.getStatus();
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
                                        if (wlm.getWorkerPool(wpc).isFullyScaled()) {
                                            data.put(modelName, new StatusResponse("Healthy"));
                                        } else {
                                            data.put(modelName, new StatusResponse("Unhealthy"));
                                        }
                                        break;
                                }
                            }
                        }
                    }
                    Map<String, Object> modelInfos = new LinkedHashMap<>(); // NOPMD
                    modelInfos.put("hasFailure", hasFailure);
                    modelInfos.put("hasPending", hasPending);
                    modelInfos.put("data", data);
                    return modelInfos;
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
