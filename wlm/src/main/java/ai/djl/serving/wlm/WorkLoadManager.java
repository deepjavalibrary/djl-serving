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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.serving.wlm.util.WlmCapacityException;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.wlm.util.WlmShutdownException;
import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.stream.Collectors;

/**
 * WorkLoadManager is responsible to manage the work load of worker thread. the manage scales
 * up/down the required amount of worker threads per model.
 *
 * @author erik.bamberg@web.de
 */
public class WorkLoadManager implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(WorkLoadManager.class);
    private ExecutorService threadPool;

    private ConcurrentHashMap<ModelInfo<?, ?>, WorkerPool<?, ?>> workerPools;

    /** Constructs a {@link WorkLoadManager} instance. */
    public WorkLoadManager() {
        threadPool = Executors.newCachedThreadPool();
        workerPools = new ConcurrentHashMap<>();
    }

    /**
     * Returns the workers for the specific model.
     *
     * @param <I> the model input class
     * @param <O> the model output class
     * @param modelInfo the name of the model we are looking for.
     * @return the list of workers responsible to handle predictions for this model.
     */
    public <I, O> List<WorkerThread<I, O>> getWorkers(ModelInfo<I, O> modelInfo) {
        List<WorkerThread<I, O>> list;
        WorkerPool<I, O> pool = getWorkerPoolForModel(modelInfo);
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
    public void unregisterModel(ModelInfo<?, ?> model) {
        WorkerPool<?, ?> pool = getWorkerPoolForModel(model);
        pool.scaleWorkers(null, 0, 0);
        workerPools.remove(model);
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

        WorkerPool<I, O> pool = getWorkerPoolForModel(modelInfo);
        int maxWorkers = pool.getMaxWorkers();
        if (maxWorkers == 0) {
            result.completeExceptionally(
                    new WlmShutdownException("All model workers has been shutdown: " + modelInfo));
            return result;
        }
        LinkedBlockingDeque<WorkerJob<I, O>> queue = pool.getJobQueue();
        if (!queue.offer(new WorkerJob<>(job, result))) {
            result.completeExceptionally(
                    new WlmCapacityException(
                            "Worker queue capacity exceeded for model: " + modelInfo));
            return result;
        }

        int currentWorkers = getNumRunningWorkers(modelInfo);
        if (currentWorkers == 0
                || currentWorkers < maxWorkers && queue.size() > modelInfo.getBatchSize() * 2) {
            synchronized (pool) {
                currentWorkers = getNumRunningWorkers(modelInfo); // check again
                if (currentWorkers < maxWorkers) {
                    logger.info(
                            "Scaling up workers for model {} to {} ",
                            modelInfo,
                            currentWorkers + 1);
                    pool.addThreads(1);
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
     * Returns the current number of request in the queue.
     *
     * @param modelInfo the model
     * @return the current number of request in the queue
     */
    public int getQueueLength(ModelInfo<?, ?> modelInfo) {
        WorkerPool<?, ?> pool = getWorkerPoolForModel(modelInfo);
        return pool.getJobQueue().size();
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
    public <I, O> WorkerPool<I, O> getWorkerPoolForModel(ModelInfo<I, O> modelInfo) {
        return (WorkerPool<I, O>)
                workerPools.computeIfAbsent(modelInfo, k -> new WorkerPool<>(modelInfo));
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        threadPool.shutdownNow();
        for (WorkerPool<?, ?> wp : workerPools.values()) {
            wp.close();
        }
    }

    /**
     * Manages the work load for a single model.
     *
     * @author erik.bamberg@web.de
     */
    public final class WorkerPool<I, O> implements AutoCloseable {

        private final ModelInfo<I, O> model;
        private Map<Device, WorkerPoolDevice> devices;
        private LinkedBlockingDeque<WorkerJob<I, O>> jobQueue;

        /**
         * Construct and initial data structure.
         *
         * @param model the model this WorkerPool belongs to.
         */
        public WorkerPool(ModelInfo<I, O> model) {
            this.model = model;
            devices = new ConcurrentHashMap<>();
            jobQueue = new LinkedBlockingDeque<>(model.getQueueSize());
        }

        /**
         * Returns a list of worker thread.
         *
         * @return the workers
         */
        public List<WorkerThread<I, O>> getWorkers() {
            return devices.values().stream()
                    .flatMap(d -> d.workers.stream())
                    .collect(Collectors.toList());
        }

        /**
         * Returns the {@code JobQueue} for this model.
         *
         * @return the jobQueue
         */
        public LinkedBlockingDeque<WorkerJob<I, O>> getJobQueue() {
            return jobQueue;
        }

        /**
         * Returns the minimum number of workers for a model across all devices.
         *
         * @return the minimum number of workers for a model across all devices
         */
        public int getMinWorkers() {
            return devices.values().stream().mapToInt(d -> d.minWorkers).reduce(0, Integer::sum);
        }

        /**
         * Returns the maximum number of workers for a model across all devices.
         *
         * @return the maximum number of workers for a model across all devices
         */
        public int getMaxWorkers() {
            return devices.values().stream().mapToInt(d -> d.maxWorkers).reduce(0, Integer::sum);
        }

        /**
         * Sets new worker capcities for this model.
         *
         * @param device the device for the model or cpu if null
         * @param newMinWorkers minimum amount of workers.
         * @param newMaxWorkers maximum amount of workers.
         * @return this {@link ModelInfo}
         */
        public WorkerPool<I, O> scaleWorkers(Device device, int newMinWorkers, int newMaxWorkers) {
            synchronized (model) {
                try {
                    model.load(device);
                } catch (ModelException | IOException e) {
                    throw new CompletionException(e);
                }
                if (model.getStatus() != ModelInfo.Status.READY) {
                    logger.warn("Cannot scale workers while model is not READY: {}", model);
                    return this;
                }
                device = model.withDefaultDevice(device);
                WlmConfigManager configManager = WlmConfigManager.getInstance();
                newMaxWorkers = configManager.getDefaultMaxWorkers(model, device, newMaxWorkers);
                newMinWorkers =
                        configManager.getDefaultMinWorkers(
                                model, device, newMinWorkers, newMaxWorkers);

                WorkerPoolDevice wpd = new WorkerPoolDevice(device, newMinWorkers, newMaxWorkers);
                devices.put(device, wpd);

                cleanup();

                List<WorkerThread<I, O>> threads;

                threads = getWorkers();
                List<WorkerThread<I, O>> fixedPoolThread =
                        threads.stream()
                                .filter(WorkerThread::isFixPoolThread)
                                .collect(Collectors.toList());

                int numberOfCurrentFixedWorkers = fixedPoolThread.size();

                if (numberOfCurrentFixedWorkers < newMinWorkers) {
                    // scale up the fixed pool
                    wpd.addThreads(newMinWorkers - numberOfCurrentFixedWorkers, true);
                } else {
                    // scale down the fixed pool
                    fixedPoolThread
                            .subList(newMinWorkers, numberOfCurrentFixedWorkers)
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

        /**
         * Returns the {@link WorkerPoolDevice} for a particular {@link Device}.
         *
         * @param device the device for a worker pool
         * @return the {@link WorkerPoolDevice} or null if device is not used
         */
        public WorkerPoolDevice forDevice(Device device) {
            return devices.get(device);
        }

        /**
         * Logs the current state of this {@code WorkerPool} when level "Debug" is enabled.
         *
         * <p>Logs all thread-ids in the pool.
         */
        public void log() {
            if (logger.isDebugEnabled()) {
                StringBuffer buf = new StringBuffer();
                getWorkers()
                        .forEach(
                                w -> {
                                    buf.append(w.getWorkerId());
                                    if (w.isFixPoolThread()) {
                                        buf.append("-fixedPool\n");
                                    } else {
                                        buf.append("-tmpPool\n");
                                    }
                                });
                logger.debug("worker pool for model {}:\n {}", model, buf);
            }
        }

        /** removes all stopped workers and workers in state error from the pool. */
        public void cleanup() {
            for (WorkerPoolDevice wpd : devices.values()) {
                wpd.workers.removeIf(
                        t ->
                                t.getState() == WorkerState.WORKER_STOPPED
                                        || t.getState() == WorkerState.WORKER_ERROR);
            }
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            model.close();
            for (WorkerPoolDevice wpd : devices.values()) {
                for (WorkerThread<I, O> worker : wpd.workers) {
                    worker.shutdown(WorkerState.WORKER_STOPPED);
                }
            }
            for (WorkerJob<I, O> wj : jobQueue) {
                wj.getFuture().cancel(true);
            }
        }

        /**
         * Adds temporary threads across existing devices.
         *
         * <p>Only supports temporary threads because permanent threads are managed per-device, so
         * it doesn't need a multi-device version.
         *
         * @param count number of threads to add
         */
        private void addThreads(int count) {
            // Add threads to devices in a random order for now
            List<WorkerPoolDevice> shuffled = new ArrayList<>(devices.values());
            Collections.shuffle(shuffled);

            for (WorkerPoolDevice wpd : devices.values()) {
                int toAdd = Math.min(count, wpd.getMaxWorkers() - wpd.workers.size());
                wpd.addThreads(toAdd, false);
                count -= toAdd;
                if (count == 0) {
                    return;
                }
            }
        }

        /**
         * The {@link WorkerPoolDevice} manages the {@link WorkerPool} for a particular {@link
         * Device}.
         */
        public final class WorkerPoolDevice {

            private Device device;
            private int minWorkers;
            private int maxWorkers;
            private List<WorkerThread<I, O>> workers;

            private WorkerPoolDevice(Device device, int minWorkers, int maxWorkers) {
                this.device = device;
                this.minWorkers = minWorkers;
                this.maxWorkers = maxWorkers;
                workers = new CopyOnWriteArrayList<>();
            }

            /**
             * Returns the min number of workers for the model and device.
             *
             * @return the min number of workers for the model and device
             */
            public int getMinWorkers() {
                return minWorkers;
            }

            /**
             * Returns the max number of workers for the model and device.
             *
             * @return the max number of workers for the model and device
             */
            public int getMaxWorkers() {
                return maxWorkers;
            }

            private void addThreads(int count, boolean permanent) {

                for (int i = 0; i < count; ++i) {

                    WorkerThread<I, O> thread =
                            WorkerThread.builder(model.getInputClass(), model.getOutputClass())
                                    .setModel(model)
                                    .setDevice(device)
                                    .setJobQueue(jobQueue)
                                    .optFixPoolThread(permanent)
                                    .build();

                    workers.add(thread);
                    threadPool.submit(thread);
                }
            }
        }
    }
}
