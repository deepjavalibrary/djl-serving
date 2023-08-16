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

import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.ExecutorService;

/** The {@link WorkerGroup} manages the {@link WorkerPool} for a particular {@link Device}. */
public class WorkerGroup<I, O> {

    private WorkerPool<I, O> workerPool;
    private Device device;
    private int minWorkers;
    int maxWorkers;
    List<WorkerThread<I, O>> workers;

    WorkerGroup(WorkerPool<I, O> workerPool, Device device) {
        this.workerPool = workerPool;
        this.device = device;
        workers = new CopyOnWriteArrayList<>();
        ModelInfo<I, O> model = workerPool.getModel();

        // Default workers from model, may be overridden by configureWorkers on init or scale
        minWorkers = model.getMinWorkers(device);
        maxWorkers = model.getMaxWorkers(device);
        minWorkers = Math.min(minWorkers, maxWorkers);
    }

    /**
     * Returns the device of the worker group.
     *
     * @return the device of the worker group
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns a list of workers.
     *
     * @return a list of workers
     */
    public List<WorkerThread<I, O>> getWorkers() {
        return workers;
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

    /**
     * Configures minimum and maximum number of workers.
     *
     * @param minWorkers the minimum number of workers
     * @param maxWorkers the maximum number of workers
     */
    public void configureWorkers(int minWorkers, int maxWorkers) {
        if (minWorkers >= 0) {
            this.minWorkers = minWorkers;
        }
        if (maxWorkers >= 0) {
            this.maxWorkers = maxWorkers;
        }
    }

    void addThreads(int count, boolean permanent) {
        ModelInfo<I, O> model = workerPool.getModel();
        ExecutorService threadPool = workerPool.getThreadPool();
        for (int i = 0; i < count; ++i) {
            WorkerThread<I, O> thread =
                    WorkerThread.builder(model)
                            .setDevice(device)
                            .setJobQueue(workerPool.getJobQueue())
                            .optFixPoolThread(permanent)
                            .build();

            workers.add(thread);
            threadPool.submit(thread);
        }
    }
}
