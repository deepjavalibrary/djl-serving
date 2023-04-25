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
package ai.djl.serving.wlm;

import ai.djl.serving.wlm.util.WlmCapacityException;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.wlm.util.WlmShutdownException;
import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * WorkLoadManager is responsible to manage the work load of worker thread. the manage scales
 * up/down the required amount of worker threads per model.
 *
 * @author erik.bamberg@web.de
 */
public class WorkLoadManager {

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    private ExecutorService threadPool;
    private ConcurrentHashMap<ModelInfo<?, ?>, WorkerPool<?, ?>> workerPools;

    /** Constructs a {@link WorkLoadManager} instance. */
    public WorkLoadManager() {
        threadPool = Executors.newCachedThreadPool();
        workerPools = new ConcurrentHashMap<>();
    }

    /**
     * Registers a model and returns the {@link WorkerPool} for it.
     *
     * <p>This operation is idempotent and will return the existing workerpool if the model was
     * already registered.
     *
     * @param <I> the model input class
     * @param <O> the model output class
     * @param modelInfo the model to create the worker pool for
     * @return the {@link WorkerPool}
     */
    @SuppressWarnings("unchecked")
    public <I, O> WorkerPool<I, O> registerModel(ModelInfo<I, O> modelInfo) {
        return (WorkerPool<I, O>)
                workerPools.computeIfAbsent(
                        modelInfo, k -> new WorkerPool<>(modelInfo, threadPool));
    }

    /**
     * Removes a model from management.
     *
     * @param model the model to remove
     */
    public void unregisterModel(ModelInfo<?, ?> model) {
        WorkerPool<?, ?> pool = getWorkerPool(model);
        if (pool.decreaseRef() <= 0) {
            logger.info("Unloading model: {}", model);
            pool.shutdownWorkers();
            workerPools.remove(model);
        }
    }

    /**
     * Adds an inference job to the job queue of the next free worker. scales up worker if
     * necessary.
     *
     * @param <I> the model input class
     * @param <O> the model output class
     * @param job an inference job to be executed.
     * @return {@code true} if submit success, false otherwise.
     */
    public <I, O> CompletableFuture<O> runJob(Job<I, O> job) {
        CompletableFuture<O> result = new CompletableFuture<>();
        ModelInfo<I, O> modelInfo = job.getModel();
        if (modelInfo.getStatus() != ModelInfo.Status.READY) {
            result.completeExceptionally(
                    new WlmException("Model is not ready: " + modelInfo.getStatus()));
            return result;
        }

        WorkerPool<I, O> pool = getWorkerPool(modelInfo);
        int maxWorkers = pool.getMaxWorkers();
        if (maxWorkers == 0) {
            result.completeExceptionally(
                    new WlmShutdownException("All model workers has been shutdown: " + modelInfo));
            return result;
        }
        LinkedBlockingDeque<WorkerJob<I, O>> queue = pool.getJobQueue();
        if ((queue.remainingCapacity() == 1 && pool.isAllWorkerBusy())
                || !queue.offer(new WorkerJob<>(job, result))) {
            result.completeExceptionally(
                    new WlmCapacityException(
                            "Worker queue capacity exceeded for model: " + modelInfo));
            scaleUp(pool, modelInfo, maxWorkers);
            return result;
        }

        int currentWorkers = getNumRunningWorkers(modelInfo);
        if (currentWorkers == 0
                || currentWorkers < maxWorkers && queue.size() > modelInfo.getBatchSize() * 2) {
            scaleUp(pool, modelInfo, maxWorkers);
        }
        return result;
    }

    private <I, O> void scaleUp(WorkerPool<I, O> pool, ModelInfo<I, O> modelInfo, int maxWorkers) {
        synchronized (pool) {
            int currentWorkers = getNumRunningWorkers(modelInfo); // check again
            if (currentWorkers < maxWorkers) {
                logger.info(
                        "Scaling up workers for model {} to {} ", modelInfo, currentWorkers + 1);
                pool.addThreads();
            }
        }
    }

    /**
     * Returns the number of running workers of a model. running workers are workers which are not
     * stopped, in error or scheduled to scale down.
     *
     * @param modelInfo the model we are interested in.
     * @return number of running workers.
     */
    public int getNumRunningWorkers(ModelInfo<?, ?> modelInfo) {
        int numWorking = 0;
        WorkerPool<?, ?> pool = workerPools.get(modelInfo);
        if (pool != null) {
            pool.cleanup();
            List<? extends WorkerThread<?, ?>> threads = pool.getWorkers();
            for (WorkerThread<?, ?> thread : threads) {
                if ((thread.getState() != WorkerState.WORKER_STOPPED)
                        && (thread.getState() != WorkerState.WORKER_ERROR)
                        && (thread.getState() != WorkerState.WORKER_SCALED_DOWN)) {
                    ++numWorking;
                }
            }
        }
        return numWorking;
    }

    /**
     * Returns the {@link WorkerPool} for a model.
     *
     * @param <I> the model input class
     * @param <O> the model output class
     * @param modelInfo the model to get the worker pool for
     * @return the {@link WorkerPool}
     */
    @SuppressWarnings("unchecked")
    public <I, O> WorkerPool<I, O> getWorkerPool(ModelInfo<I, O> modelInfo) {
        return (WorkerPool<I, O>) workerPools.get(modelInfo);
    }

    /** Close all models related to the {@code WorkloadManager}. */
    public void close() {
        threadPool.shutdownNow();
        for (WorkerPool<?, ?> wp : workerPools.values()) {
            wp.shutdown();
        }
        workerPools.clear();
    }
}
