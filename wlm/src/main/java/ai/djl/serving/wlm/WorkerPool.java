/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;

/**
 * Manages the work load for a single model.
 *
 * @author erik.bamberg@web.de
 */
public class WorkerPool<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(WorkerPool.class);

    private final WorkerPoolConfig<I, O> wpc;
    private ExecutorService threadPool;
    private Map<Device, WorkerGroup<I, O>> workerGroups;
    private BlockingQueue<WorkerJob<I, O>> jobQueue;
    private AtomicInteger refCnt;

    /**
     * Construct and initial data structure.
     *
     * @param wpc the model this WorkerPool belongs to
     * @param threadPool the thread pool executor
     */
    WorkerPool(WorkerPoolConfig<I, O> wpc, ExecutorService threadPool) {
        this.wpc = wpc;
        this.threadPool = threadPool;
        workerGroups = new ConcurrentHashMap<>();
        refCnt = new AtomicInteger(1);
    }

    /** Increases the reference count. */
    public void increaseRef() {
        refCnt.incrementAndGet();
    }

    /**
     * Decrease the reference count and return the current count.
     *
     * @return the current count
     */
    public int decreaseRef() {
        return refCnt.decrementAndGet();
    }

    /**
     * Returns the model of the worker pool.
     *
     * @return the model of the worker pool
     */
    public WorkerPoolConfig<I, O> getWpc() {
        return wpc;
    }

    ExecutorService getThreadPool() {
        return threadPool;
    }

    /**
     * Returns a map of {@code WorkerGroup}.
     *
     * @return a map of {@code WorkerGroup}
     */
    public Map<Device, WorkerGroup<I, O>> getWorkerGroups() {
        return workerGroups;
    }

    /**
     * Returns a list of worker thread.
     *
     * @return the workers
     */
    public List<WorkerThread<I, O>> getWorkers() {
        return workerGroups.values().stream()
                .flatMap(g -> g.workers.stream())
                .collect(Collectors.toList());
    }

    /**
     * Returns the {@code JobQueue} for this model.
     *
     * @return the jobQueue
     */
    public BlockingQueue<WorkerJob<I, O>> getJobQueue() {
        return jobQueue;
    }

    /**
     * Returns the maximum number of workers for a model across all devices.
     *
     * @return the maximum number of workers for a model across all devices
     */
    public int getMaxWorkers() {
        return workerGroups.values().stream().mapToInt(g -> g.maxWorkers).reduce(0, Integer::sum);
    }

    /**
     * Return if all workers died.
     *
     * @return true if all workers died
     */
    public boolean isAllWorkerDied() {
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            for (WorkerThread<?, ?> thread : group.getWorkers()) {
                if (thread.isRunning()) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Returns {@code true} if all workers are busy.
     *
     * @return {@code true} if all workers are busy
     */
    public boolean isAllWorkerBusy() {
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            for (WorkerThread<?, ?> thread : group.getWorkers()) {
                if (thread.getState() == WorkerState.WORKER_STARTED) {
                    return false;
                }
            }
        }
        return true;
    }

    /**
     * Returns if the worker groups is fully scaled.
     *
     * @return true if the worker groups is fully scaled
     */
    public boolean isFullyScaled() {
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            if (group.getMinWorkers() > group.getWorkers().size()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Initializes new worker capacities for this model.
     *
     * @param deviceName the device for the model, null for default devices
     */
    public void initWorkers(String deviceName) {
        initWorkers(deviceName, -1, -1);
    }

    /**
     * Initializes new worker capacities for this model.
     *
     * @param deviceName the device for the model, null for default devices
     * @param minWorkers minimum amount of workers (-1 for model default).
     * @param maxWorkers maximum amount of workers (-1 for model default).
     */
    public void initWorkers(String deviceName, int minWorkers, int maxWorkers) {
        Device device;
        synchronized (wpc) {
            try {
                wpc.initialize();
                device = wpc.withDefaultDevice(deviceName);
                logger.info("loading model {} on {} ...", wpc, device);
                wpc.load(device);
            } catch (ModelException | IOException e) {
                throw new CompletionException(e);
            }
            if (wpc.getStatus() != WorkerPoolConfig.Status.READY) {
                logger.warn("Cannot scale workers while model is not READY: {}", wpc);
            }
        }

        // jobQueue should be initialized after model is configure
        if (jobQueue == null) {
            jobQueue = new PriorityBlockingQueue<>(wpc.getQueueSize());
        }
        cleanup();

        WorkerGroup<I, O> group =
                workerGroups.computeIfAbsent(device, d -> new WorkerGroup<>(this, d));
        group.configureWorkers(minWorkers, maxWorkers);
        doScaleWorker(group);
        log();
    }

    /**
     * Sets new worker capacities for this model.
     *
     * @param deviceName the device for the model, null for all loaded devices
     * @param minWorkers minimum amount of workers.
     * @param maxWorkers maximum amount of workers.
     */
    public void scaleWorkers(String deviceName, int minWorkers, int maxWorkers) {
        if (deviceName != null) {
            // if the model has not been loaded on device, this will load the model
            initWorkers(deviceName, minWorkers, maxWorkers);
            return;
        }

        cleanup();

        // scale for all devices
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            group.configureWorkers(minWorkers, maxWorkers);
            doScaleWorker(group);
        }
        log();
    }

    private void doScaleWorker(WorkerGroup<I, O> group) {
        int minWorkers = group.getMinWorkers();
        List<WorkerThread<I, O>> fixedPoolThreads = new ArrayList<>();
        for (WorkerThread<I, O> threads : group.getWorkers()) {
            if (threads.isFixPoolThread()) {
                fixedPoolThreads.add(threads);
            }
        }
        int activeThreads = fixedPoolThreads.size();
        if (activeThreads < minWorkers) {
            // scale up the fixed pool
            logger.info(
                    "scaling up min workers by {} (from {} to {}) workers. Total range is min {} to"
                            + " max {}",
                    minWorkers - activeThreads,
                    activeThreads,
                    minWorkers,
                    minWorkers,
                    group.getMaxWorkers());
            group.addThreads(minWorkers - activeThreads, true);
        } else {
            // scale down the fixed pool
            logger.info("scale down from workers {} to {}", activeThreads, minWorkers);
            fixedPoolThreads
                    .subList(minWorkers, activeThreads)
                    .forEach(t -> t.shutdown(WorkerState.WORKER_SCALED_DOWN));
        }
    }

    /** Shutdown all works. */
    public void shutdownWorkers() {
        synchronized (wpc) {
            List<WorkerThread<I, O>> threads = getWorkers();
            for (WorkerThread<I, O> thread : threads) {
                thread.shutdown(WorkerState.WORKER_SCALED_DOWN);
            }
            threads.clear();
        }
    }

    /** removes all stopped workers and workers in state error from the pool. */
    public void cleanup() {
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            group.workers.removeIf(WorkerThread::isStale);
        }
    }

    /** Shuts down all the worker threads in the work pool. */
    public void shutdown() {
        wpc.close();
        for (WorkerGroup<I, O> group : workerGroups.values()) {
            for (WorkerThread<I, O> worker : group.workers) {
                worker.shutdown(WorkerState.WORKER_STOPPED);
            }
        }
        workerGroups.clear();
        if (jobQueue != null) {
            for (WorkerJob<I, O> wj : jobQueue) {
                wj.getFuture().cancel(true);
            }
        }
    }

    /**
     * Adds temporary threads across existing devices.
     *
     * <p>Only supports temporary threads because permanent threads are managed per-device, so it
     * doesn't need a multi-device version.
     */
    void addThreads() {
        // Add threads to devices which has most room to grow
        List<WorkerGroup<I, O>> sorted = new ArrayList<>(workerGroups.values());
        if (sorted.isEmpty()) {
            logger.warn("No worker pool available.");
            return;
        }
        sorted.sort(Comparator.comparingInt(p -> p.getMaxWorkers() - p.workers.size()));

        WorkerGroup<I, O> group = sorted.get(sorted.size() - 1);
        if (group.getMaxWorkers() > group.workers.size()) {
            group.addThreads(1, false);
        }
    }

    /**
     * Logs the current state of this {@code WorkerPool} when level "Debug" is enabled.
     *
     * <p>Logs all thread-ids in the pool.
     */
    private void log() {
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
            logger.debug("worker pool for model {}:\n {}", wpc, buf);
        }
    }
}
