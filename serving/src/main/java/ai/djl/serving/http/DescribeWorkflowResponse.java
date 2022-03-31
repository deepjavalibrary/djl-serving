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
package ai.djl.serving.http;

import ai.djl.Device;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

/** A class that holds information about workflow status. */
public class DescribeWorkflowResponse {

    private String workflowName;
    private String workflowUrl;
    private int minWorkers;
    private int maxWorkers;
    private int batchSize;
    private int maxBatchDelay;
    private int maxIdleTime;
    private int queueLength;
    private String status;
    private boolean loadedAtStartup;

    private List<Worker> workers;

    /** Constructs a {@code DescribeWorkflowResponse} instance. */
    public DescribeWorkflowResponse() {
        workers = new ArrayList<>();
    }

    /**
     * Returns the workflow name.
     *
     * @return the workflow name
     */
    public String getWorkflowName() {
        return workflowName;
    }

    /**
     * Sets the workflow name.
     *
     * @param workflowName the workflow name
     */
    public void setWorkflowName(String workflowName) {
        this.workflowName = workflowName;
    }

    /**
     * Returns if the workflows was loaded at startup.
     *
     * @return {@code true} if the workflows was loaded at startup
     */
    public boolean isLoadedAtStartup() {
        return loadedAtStartup;
    }

    /**
     * Sets the load at startup status.
     *
     * @param loadedAtStartup {@code true} if the workflows was loaded at startup
     */
    public void setLoadedAtStartup(boolean loadedAtStartup) {
        this.loadedAtStartup = loadedAtStartup;
    }

    /**
     * Returns the workflow URL.
     *
     * @return the workflow URL
     */
    public String getWorkflowUrl() {
        return workflowUrl;
    }

    /**
     * Sets the workflow URL.
     *
     * @param workflowUrl the workflow URL
     */
    public void setWorkflowUrl(String workflowUrl) {
        this.workflowUrl = workflowUrl;
    }

    /**
     * Returns the desired minimum number of workers.
     *
     * @return the desired minimum number of workers
     */
    public int getMinWorkers() {
        return minWorkers;
    }

    /**
     * Sets the desired minimum number of workers.
     *
     * @param minWorkers the desired minimum number of workers
     */
    public void setMinWorkers(int minWorkers) {
        this.minWorkers = minWorkers;
    }

    /**
     * Returns the desired maximum number of workers.
     *
     * @return the desired maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
    }

    /**
     * Sets the desired maximum number of workers.
     *
     * @param maxWorkers the desired maximum number of workers
     */
    public void setMaxWorkers(int maxWorkers) {
        this.maxWorkers = maxWorkers;
    }

    /**
     * Returns the batch size.
     *
     * @return the batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the batch size.
     *
     * @param batchSize the batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    /**
     * Sets the maximum delay in milliseconds to aggregate a batch.
     *
     * @param maxBatchDelay the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    /**
     * Returns the number of request in the queue.
     *
     * @return the number of request in the queue
     */
    public int getQueueLength() {
        return queueLength;
    }

    /**
     * Sets the number of request in the queue.
     *
     * @param queueLength the number of request in the queue
     */
    public void setQueueLength(int queueLength) {
        this.queueLength = queueLength;
    }

    /**
     * Returns the workflow's status.
     *
     * @return the workflow's status
     */
    public String getStatus() {
        return status;
    }

    /**
     * Sets the workflow's status.
     *
     * @param status the workflow's status
     */
    public void setStatus(String status) {
        this.status = status;
    }

    /**
     * Sets the max idle time for worker threads.
     *
     * @param maxIdleTime the time a worker thread can be idle before scaling down.
     */
    public void setMaxIdleTime(int maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
    }

    /**
     * Returns the maximum idle time for worker threads.
     *
     * @return the maxIdleTime
     */
    public int getMaxIdleTime() {
        return maxIdleTime;
    }

    /**
     * Returns all workers information of the workflow.
     *
     * @return all workers information of the workflow
     */
    public List<Worker> getWorkers() {
        return workers;
    }

    /**
     * Adds worker to the worker list.
     *
     * @param version the workflow version
     * @param id the worker's ID
     * @param startTime the worker's start time
     * @param isRunning {@code true} if worker is running
     * @param device the device assigned to the worker
     */
    public void addWorker(
            String version, int id, long startTime, boolean isRunning, Device device) {
        Worker worker = new Worker();
        worker.setVersion(version);
        worker.setId(id);
        worker.setStartTime(new Date(startTime));
        worker.setStatus(isRunning ? "READY" : "UNLOADING");
        worker.setDevice(device);
        workers.add(worker);
    }

    /** A class that holds workers information. */
    public static final class Worker {

        private String version;
        private int id;
        private Date startTime;
        private String status;
        private Device device;

        /**
         * Returns the model version.
         *
         * @return the model version
         */
        public String getVersion() {
            return version;
        }

        /**
         * Sets the model version.
         *
         * @param version the model version
         */
        public void setVersion(String version) {
            this.version = version;
        }

        /**
         * Returns the worker's ID.
         *
         * @return the worker's ID
         */
        public int getId() {
            return id;
        }

        /**
         * Sets the worker's ID.
         *
         * @param id the workers ID
         */
        public void setId(int id) {
            this.id = id;
        }

        /**
         * Returns the worker's start time.
         *
         * @return the worker's start time
         */
        public Date getStartTime() {
            return startTime;
        }

        /**
         * Sets the worker's start time.
         *
         * @param startTime the worker's start time
         */
        public void setStartTime(Date startTime) {
            this.startTime = startTime;
        }

        /**
         * Returns the worker's status.
         *
         * @return the worker's status
         */
        public String getStatus() {
            return status;
        }

        /**
         * Sets the worker's status.
         *
         * @param status the worker's status
         */
        public void setStatus(String status) {
            this.status = status;
        }

        /**
         * Return if the worker using GPU.
         *
         * @return {@code true} if the worker using GPU
         */
        public boolean isGpu() {
            return device.isGpu();
        }

        /**
         * Sets the worker device.
         *
         * @param device the worker device
         */
        public void setDevice(Device device) {
            this.device = device;
        }
    }
}
