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

import ai.djl.modality.Output;
import ai.djl.ndarray.NDManager;
import ai.djl.serving.wlm.util.WlmCapacityException;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.wlm.util.WlmShutdownException;
import ai.djl.serving.wlm.util.WorkerJob;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * WorkLoadManager is responsible to manage the work load of worker thread. the manage scales
 * up/down the required amount of worker threads per model.
 *
 * @author erik.bamberg@web.de
 */
public class WorkLoadManager {

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);
    private ExecutorService threadPool;

    private ConcurrentHashMap<ModelInfo, WorkerPool> workerPools;

    /** Constructs a {@link WorkLoadManager} instance. */
    public WorkLoadManager() {
        threadPool = Executors.newCachedThreadPool();
        workerPools = new ConcurrentHashMap<>();
    }

    /**
     * Returns the workers for the specific model.
     *
     * @param modelInfo the name of the model we are looking for.
     * @return the list of workers responsible to handle predictions for this model.
     */
    public List<WorkerThread> getWorkers(ModelInfo modelInfo) {
        List<WorkerThread> list;
        WorkerPool pool = workerPools.get(modelInfo);
        if (pool == null) {
            list = Collections.emptyList();
        } else {
            list = pool.getWorkers();
            if (list == null) {
                list = Collections.emptyList();
            }
        }
        return list;
    }

    /**
     * Removes a model from management.
     *
     * @param model the model to remove
     */
    public void unregisterModel(ModelInfo model) {
        WorkerPool pool = getWorkerPoolForModel(model);
        pool.scaleWorkers(null, 0, 0);
        workerPools.remove(model);
    }

    /**
     * Adds an inference job to the job queue of the next free worker. scales up worker if
     * necessary.
     *
     * @param job an inference job to be executed.
     * @return {@code true} if submit success, false otherwise.
     */
    public CompletableFuture<Output> runJob(Job job) {
        CompletableFuture<Output> result = new CompletableFuture<>();
        ModelInfo modelInfo = job.getModel();
        WorkerPool pool = getWorkerPoolForModel(modelInfo);
        int maxWorkers = pool.getMaxWorkers();
        if (maxWorkers == 0) {
            result.completeExceptionally(
                    new WlmShutdownException(
                            "All model workers has been shutdown: " + modelInfo.getModelName()));
            return result;
        }
        LinkedBlockingDeque<WorkerJob> queue = pool.getJobQueue();
        if (!queue.offer(new WorkerJob(job, result))) {
            result.completeExceptionally(
                    new WlmCapacityException(
                            "Worker queue capacity exceeded for model: "
                                    + modelInfo.getModelName()));
            return result;
        }

        int currentWorkers = getNumRunningWorkers(modelInfo);
        if (currentWorkers == 0
                || currentWorkers < maxWorkers && queue.size() > modelInfo.getBatchSize() * 2) {
            synchronized (modelInfo.getModel()) {
                currentWorkers = getNumRunningWorkers(modelInfo); // check again
                if (currentWorkers < maxWorkers) {
                    logger.info(
                            "Scaling up workers for model {} to {} ",
                            modelInfo,
                            currentWorkers + 1);
                    pool.addThreads(modelInfo, 1, false);
                }
            }
        }
        return result;
    }

    /**
     * Returns the number of running workers of a model. running workers are workers which are not
     * stopped, in error or scheduled to scale down.
     *
     * @param modelInfo the model we are interested in.
     * @return number of running workers.
     */
    public int getNumRunningWorkers(ModelInfo modelInfo) {
        int numWorking = 0;
        WorkerPool pool = workerPools.get(modelInfo);
        if (pool != null) {
            pool.cleanup();
            List<WorkerThread> threads = pool.getWorkers();
            for (WorkerThread thread : threads) {
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
     * Returns the current number of request in the queue.
     *
     * @param modelInfo the model
     * @return the current number of request in the queue
     */
    public int getQueueLength(ModelInfo modelInfo) {
        WorkerPool pool = getWorkerPoolForModel(modelInfo);
        return pool.getJobQueue().size();
    }

    /**
     * Returns the {@link WorkerPool} for a model.
     *
     * @param modelInfo the model to get the worker pool for
     * @return the {@link WorkerPool}
     */
    public WorkerPool getWorkerPoolForModel(ModelInfo modelInfo) {
        return workerPools.computeIfAbsent(modelInfo, k -> new WorkerPool(modelInfo));
    }

    /**
     * Manages the work load for a single model.
     *
     * @author erik.bamberg@web.de
     */
    public final class WorkerPool {

        private final ModelInfo model;
        private List<WorkerThread> workers;
        private LinkedBlockingDeque<WorkerJob> jobQueue;
        private int minWorkers;
        private int maxWorkers;

        /**
         * Construct and initial data structure.
         *
         * @param model the model this WorkerPool belongs to.
         */
        public WorkerPool(ModelInfo model) {
            this.model = model;
            workers = new CopyOnWriteArrayList<>();
            jobQueue = new LinkedBlockingDeque<>(model.getQueueSize());
        }

        /**
         * Returns a list of worker thread.
         *
         * @return the workers
         */
        public List<WorkerThread> getWorkers() {
            return workers;
        }

        /**
         * Returns the {@code JobQueue} for this model.
         *
         * @return the jobQueue
         */
        public LinkedBlockingDeque<WorkerJob> getJobQueue() {
            return jobQueue;
        }

        /**
         * Returns the minimum number of workers for a model.
         *
         * @return the minimum number of workers for a model
         */
        public int getMinWorkers() {
            return minWorkers;
        }

        /**
         * Returns the maximum number of workers for a model.
         *
         * @return the maximum number of workers for a model
         */
        public int getMaxWorkers() {
            return maxWorkers;
        }

        /**
         * Sets new worker capcities for this model.
         *
         * @param deviceName the device for the model
         * @param newMinWorkers minimum amount of workers.
         * @param newMaxWorkers maximum amount of workers.
         * @return this {@link ModelInfo}
         */
        public WorkerPool scaleWorkers(String deviceName, int newMinWorkers, int newMaxWorkers) {
            synchronized (model) {
                NDManager manager = model.getModel().getNDManager();
                WlmConfigManager configManager = WlmConfigManager.getInstance();
                maxWorkers = configManager.getDefaultWorkers(manager, deviceName, newMaxWorkers);
                minWorkers = Math.min(newMinWorkers, maxWorkers);

                cleanup();

                List<WorkerThread> threads;

                threads = getWorkers();
                List<WorkerThread> fixedPoolThread =
                        threads.stream()
                                .filter(WorkerThread::isFixPoolThread)
                                .collect(Collectors.toList());

                int numberOfCurrentFixedWorkers = fixedPoolThread.size();

                if (numberOfCurrentFixedWorkers < minWorkers) {
                    // scale up the fixed pool
                    addThreads(model, minWorkers - numberOfCurrentFixedWorkers, true);
                } else {
                    // scale down the fixed pool
                    fixedPoolThread
                            .subList(minWorkers, numberOfCurrentFixedWorkers)
                            .forEach(
                                    t -> {
                                        threads.remove(t);
                                        t.shutdown(WorkerState.WORKER_SCALED_DOWN);
                                    });
                }
                log();

                return this;
            }
        }

        private void addThreads(ModelInfo model, int count, boolean permanent) {

            for (int i = 0; i < count; ++i) {

                WorkerThread thread =
                        WorkerThread.builder()
                                .setModel(model)
                                .setJobQueue(jobQueue)
                                .optFixPoolThread(permanent)
                                .build();

                workers.add(thread);
                threadPool.submit(thread);
            }
        }

        /**
         * Logs the current state of this {@code WorkerPool} when level "Debug" is enabled.
         *
         * <p>Logs all thread-ids in the pool.
         */
        public void log() {
            if (logger.isDebugEnabled()) {
                StringBuffer buf = new StringBuffer();
                workers.forEach(
                        w -> {
                            buf.append(w.getWorkerId());
                            if (w.isFixPoolThread()) {
                                buf.append("-fixedPool\n");
                            } else {
                                buf.append("-tmpPool\n");
                            }
                        });
                logger.debug("worker pool for model {}:\n {}", model.getModelName(), buf);
            }
        }

        /** removes all stopped workers and workers in state error from the pool. */
        public void cleanup() {
            workers.removeIf(
                    t ->
                            t.getState() == WorkerState.WORKER_STOPPED
                                    || t.getState() == WorkerState.WORKER_ERROR);
        }
    }
}
