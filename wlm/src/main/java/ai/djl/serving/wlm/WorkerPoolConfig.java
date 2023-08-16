/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * A {@link WorkerPoolConfig} represents a task that could be run in the {@link WorkLoadManager}.
 *
 * <p>Each {@link WorkerThread} (also {@link WorkerPool} and {@link WorkerGroup}) focuses on
 * executing a single worker type. They contain the configuration for the thread, any persistent
 * data, and the code to run on the thread.
 *
 * @param <I> the input type
 * @param <O> the output type
 */
public abstract class WorkerPoolConfig<I, O> {

    protected transient String id;
    protected String version;
    protected String modelUrl;
    protected int queueSize;
    protected int batchSize;
    protected int maxBatchDelayMillis;
    protected int maxIdleSeconds;
    protected Integer minWorkers; // Integer so it becomes null when parsed from JSON
    protected Integer maxWorkers; // Integer so it becomes null when parsed from JSON

    /**
     * Loads the worker type to the specified device.
     *
     * @param device the device to load worker type on
     * @throws IOException if failed to read worker type file
     * @throws ModelException if failed to load the specified model
     */
    public abstract void load(Device device) throws ModelException, IOException;

    /**
     * Starts a new {@link WorkerThread} for this {@link WorkerPoolConfig}.
     *
     * @param device the device to run on
     * @return the new {@link ThreadType}
     */
    public abstract ThreadType<I, O> newThread(Device device);

    /**
     * Initialize the worker.
     *
     * @throws IOException if failed to download worker
     * @throws ModelNotFoundException if model not found
     */
    public abstract void initialize() throws IOException, ModelException;

    /** Close all loaded workers. */
    public abstract void close();

    /**
     * Returns the default device for this model if device is null.
     *
     * @param deviceName the device to use if it is not null
     * @return a non-null device
     */
    public Device withDefaultDevice(String deviceName) {
        return Device.fromName(deviceName);
    }

    /**
     * Returns the worker type loading status.
     *
     * @return the worker type loading status
     */
    public abstract Status getStatus();

    /**
     * Returns if the worker type can be load parallel on multiple devices.
     *
     * @return if the worker type can be load parallel on multiple devices
     */
    public abstract boolean isParallelLoading();

    /**
     * Returns the devices the worker type will be loaded on at startup.
     *
     * @return the devices the worker type will be loaded on at startup
     */
    public abstract String[] getLoadOnDevices();

    /**
     * Sets the worker type ID.
     *
     * @param id the worker type ID
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * Returns the worker type ID.
     *
     * @return the worker type ID
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the worker type version.
     *
     * @return the worker type version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the worker type url.
     *
     * @return the worker type url
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Sets the configured max idle time in seconds of workers.
     *
     * @param maxIdleSeconds the configured max idle time in seconds of workers
     */
    public void setMaxIdleSeconds(int maxIdleSeconds) {
        this.maxIdleSeconds = maxIdleSeconds;
    }

    /**
     * Returns the configured max idle time in seconds of workers.
     *
     * @return the max idle time in seconds
     */
    public int getMaxIdleSeconds() {
        return maxIdleSeconds;
    }

    /**
     * Sets the configured batch size.
     *
     * @param batchSize the configured batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the configured batch size.
     *
     * @return the configured batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the maximum delay in milliseconds to aggregate a batch.
     *
     * @param maxBatchDelayMillis the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelayMillis(int maxBatchDelayMillis) {
        this.maxBatchDelayMillis = maxBatchDelayMillis;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelayMillis() {
        return maxBatchDelayMillis;
    }

    /**
     * Sets the configured size of the workers queue.
     *
     * @param queueSize the configured size of the workers queue
     */
    public void setQueueSize(int queueSize) {
        this.queueSize = queueSize;
    }

    /**
     * Returns the configured size of the workers queue.
     *
     * @return requested size of the workers queue.
     */
    public int getQueueSize() {
        return queueSize;
    }

    /**
     * Sets the starting number of min workers.
     *
     * @param minWorkers Sets the starting number of min workers
     */
    public void setMinWorkers(int minWorkers) {
        if (maxWorkers != null && maxWorkers < minWorkers) {
            throw new IllegalArgumentException(
                    "The max workers for a model or worker can't be smaller than the min workers");
        }

        this.minWorkers = minWorkers;
    }

    /**
     * Returns the minimum number of workers.
     *
     * @param device the device to get the min workers for
     * @return the minimum number of workers
     */
    public int getMinWorkers(Device device) {
        return minWorkers;
    }

    /**
     * Sets the starting number of max workers.
     *
     * @param maxWorkers Sets the starting number of max workers
     */
    public void setMaxWorkers(int maxWorkers) {
        if (minWorkers != null && maxWorkers < minWorkers) {
            throw new IllegalArgumentException(
                    "The max workers for a model or worker can't be smaller than the min workers");
        }
        if (maxWorkers == 0) {
            throw new IllegalArgumentException("Models must have a maxWorkers greater than 0");
        }

        this.maxWorkers = maxWorkers;
    }

    /**
     * Returns the maximum number of workers.
     *
     * @param device the device to get the max workers for
     * @return the maximum number of workers
     */
    public int getMaxWorkers(Device device) {
        return maxWorkers;
    }

    /**
     * Sets the starting minimum and maximum number of workers.
     *
     * @param minWorkers the new minimum number of workers
     * @param maxWorkers the new maximum number of workers
     */
    public void setMinMaxWorkers(int minWorkers, int maxWorkers) {
        if (maxWorkers < minWorkers) {
            throw new IllegalArgumentException(
                    "The max workers for a model or worker can't be smaller than the min workers");
        }
        if (minWorkers == 0) {
            throw new IllegalArgumentException(
                    "Having a minWorkers of 0 is not currently supported");
        }
        if (maxWorkers == 0) {
            throw new IllegalArgumentException("Models must have a maxWorkers greater than 0");
        }

        this.minWorkers = minWorkers;
        this.maxWorkers = maxWorkers;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof WorkerPoolConfig)) {
            return false;
        }
        WorkerPoolConfig<?, ?> wpc = (WorkerPoolConfig<?, ?>) o;
        return id.equals(wpc.id) && Objects.equals(version, wpc.version);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(id, version);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return id + ':' + version + " (" + getStatus() + ')';
        }
        return id + " (" + getStatus() + ')';
    }

    /** An enum represents state of a worker type. */
    public enum Status {
        PENDING,
        READY,
        FAILED
    }

    protected abstract static class ThreadType<I, O> {
        Device device;

        protected ThreadType(Device device) {
            this.device = device;
        }

        /**
         * Runs the work on the {@link WorkerThread}.
         *
         * @param input the work input
         * @return the computed output
         * @throws TranslateException if it failed to compute
         */
        public abstract List<O> run(List<I> input) throws TranslateException;

        /** Closes the thread type and frees any resources. */
        public abstract void close();
    }
}
