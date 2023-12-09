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
import java.util.Map.Entry;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;

/**
 * WorkLoadManager is responsible to manage the work load of worker thread. the manage scales
 * up/down the required amount of worker threads per wpc.
 *
 * @author erik.bamberg@web.de
 */
public class WorkLoadManager {

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);

    private ExecutorService threadPool;
    private ConcurrentHashMap<WorkerPoolConfig<?, ?>, WorkerPool<?, ?>> workerPools;

    /** Constructs a {@link WorkLoadManager} instance. */
    public WorkLoadManager() {
        threadPool =
                Executors.newCachedThreadPool(
                        r -> {
                            Thread t = Executors.defaultThreadFactory().newThread(r);
                            t.setDaemon(true);
                            return t;
                        });
        workerPools = new ConcurrentHashMap<>();
    }

    /**
     * Registers a {@link WorkerPool} (model).
     *
     * <p>This operation is idempotent and will return the existing workerpool if the wpc was
     * already registered.
     *
     * @param <I> the wpc input class
     * @param <O> the wpc output class
     * @param wpc the wpc to create the worker pool for
     * @return the {@link WorkerPool}
     */
    @SuppressWarnings("unchecked")
    public <I, O> WorkerPool<I, O> registerWorkerPool(WorkerPoolConfig<I, O> wpc) {
        return (WorkerPool<I, O>)
                workerPools.computeIfAbsent(wpc, k -> new WorkerPool<>(wpc, threadPool));
    }

    /**
     * Removes a worker pool from management.
     *
     * @param wpc the wpc to remove
     */
    public void unregisterWorkerPool(WorkerPoolConfig<?, ?> wpc) {
        WorkerPool<?, ?> pool = getWorkerPool(wpc);
        if (pool.decreaseRef() <= 0) {
            logger.info("Unloading model: {}", wpc);
            pool.shutdownWorkers();
            workerPools.remove(wpc);
        }
    }

    /**
     * Adds an inference job to the job queue of the next free worker. scales up worker if
     * necessary.
     *
     * @param <I> the wpc input class
     * @param <O> the wpc output class
     * @param job an inference job to be executed.
     * @return {@code true} if submit success, false otherwise.
     */
    public <I, O> CompletableFuture<O> runJob(Job<I, O> job) {
        CompletableFuture<O> result = new CompletableFuture<>();
        WorkerPoolConfig<I, O> wpc = job.getWpc();
        if (wpc.getStatus() != WorkerPoolConfig.Status.READY) {
            result.completeExceptionally(new WlmException("Model is not ready: " + wpc));
            return result;
        }

        WorkerPool<I, O> pool = getWorkerPool(wpc);
        int maxWorkers = pool.getMaxWorkers();
        if (maxWorkers == 0) {
            result.completeExceptionally(
                    new WlmShutdownException("All model workers has been shutdown: " + wpc));
            return result;
        }
        LinkedBlockingDeque<WorkerJob<I, O>> queue = pool.getJobQueue();
        if ((queue.remainingCapacity() == 1 && pool.isAllWorkerBusy())
                || pool.isAllWorkerDied()
                || !queue.offer(new WorkerJob<>(job, result))) {
            result.completeExceptionally(
                    new WlmCapacityException("Worker queue capacity exceeded for model: " + wpc));
            scaleUp(pool, wpc, maxWorkers);
            return result;
        }

        int currentWorkers = getNumRunningWorkers(wpc);
        if (currentWorkers == 0
                || currentWorkers < maxWorkers && queue.size() > wpc.getBatchSize() * 2) {
            scaleUp(pool, wpc, maxWorkers);
        }
        return result;
    }

    private <I, O> void scaleUp(WorkerPool<I, O> pool, WorkerPoolConfig<I, O> wpc, int maxWorkers) {
        synchronized (pool) {
            int currentWorkers = getNumRunningWorkers(wpc); // check again
            if (currentWorkers < maxWorkers) {
                logger.info("Scaling up workers for model {} to {} ", wpc, currentWorkers + 1);
                pool.addThreads();
            }
        }
    }

    /**
     * Returns the number of running workers of a wpc. running workers are workers which are not
     * stopped, in error or scheduled to scale down.
     *
     * @param wpc the wpc we are interested in.
     * @return number of running workers.
     */
    public int getNumRunningWorkers(WorkerPoolConfig<?, ?> wpc) {
        int numWorking = 0;
        WorkerPool<?, ?> pool = workerPools.get(wpc);
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
     * Returns the {@link WorkerPool} for a wpc.
     *
     * @param <I> the wpc class
     * @param <O> the wpc class
     * @param id the wpc id
     * @return the {@link WorkerPool}
     */
    @SuppressWarnings("unchecked")
    public <I, O> WorkerPool<I, O> getWorkerPoolById(String id) {
        for (Entry<WorkerPoolConfig<?, ?>, WorkerPool<?, ?>> wp : workerPools.entrySet()) {
            if (id.equals(wp.getKey().getId())) {
                return (WorkerPool<I, O>) wp.getValue();
            }
        }
        return null;
    }

    /**
     * Returns the {@link WorkerPool} for a model.
     *
     * @param <I> the wpc input class
     * @param <O> the wpc output class
     * @param wpc the worker type to get the worker pool for
     * @return the {@link WorkerPool}
     */
    @SuppressWarnings("unchecked")
    public <I, O> WorkerPool<I, O> getWorkerPool(WorkerPoolConfig<I, O> wpc) {
        return (WorkerPool<I, O>) workerPools.get(wpc);
    }

    /** Close all wpcs related to the {@code WorkloadManager}. */
    public void close() {
        threadPool.shutdownNow();
        for (WorkerPool<?, ?> wp : workerPools.values()) {
            wp.shutdown();
        }
        workerPools.clear();
    }
}
