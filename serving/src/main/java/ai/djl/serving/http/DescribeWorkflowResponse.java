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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerGroup;
import ai.djl.serving.wlm.WorkerPool;
import ai.djl.serving.wlm.WorkerThread;

import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Set;

/** A class that holds information about workflow status. */
public class DescribeWorkflowResponse {

    private String workflowName;
    private String version;
    private List<Model> models;

    /**
     * Constructs a new {@code DescribeWorkflowResponse} instance.
     *
     * @param workflow the workflow
     */
    public DescribeWorkflowResponse(ai.djl.serving.workflow.Workflow workflow) {
        this.workflowName = workflow.getName();
        this.version = workflow.getVersion();
        models = new ArrayList<>();

        ModelManager manager = ModelManager.getInstance();
        WorkLoadManager wlm = manager.getWorkLoadManager();
        Set<String> startupWorkflows = manager.getStartupWorkflows();

        for (ModelInfo<Input, Output> model : workflow.getModels()) {
            ModelInfo.Status status = model.getStatus();
            int activeWorker = 0;
            int targetWorker = 0;

            Model m = new Model();
            models.add(m);
            WorkerPool<Input, Output> pool = wlm.getWorkerPool(model);
            if (pool != null) {
                m.setModelName(model.getModelId());
                m.setModelUrl(model.getModelUrl());
                m.setBatchSize(model.getBatchSize());
                m.setMaxBatchDelay(model.getMaxBatchDelay());
                m.setMaxIdleTime(model.getMaxIdleTime());
                m.setQueueSize(model.getQueueSize());
                m.setRequestInQueue(pool.getJobQueue().size());
                m.setLoadedAtStartup(startupWorkflows.contains(model.getModelId()));

                for (WorkerGroup<Input, Output> group : pool.getWorkerGroups().values()) {
                    Device device = group.getDevice();
                    Group g = new Group(device, group.getMinWorkers(), group.getMaxWorkers());
                    m.addGroup(g);

                    List<WorkerThread<Input, Output>> workers = group.getWorkers();
                    activeWorker += workers.size();
                    targetWorker += group.getMinWorkers();

                    for (WorkerThread<Input, Output> worker : workers) {
                        int workerId = worker.getWorkerId();
                        long startTime = worker.getStartTime();
                        boolean isRunning = worker.isRunning();
                        g.addWorker(workerId, startTime, isRunning);
                    }
                }
            }

            if (status == ModelInfo.Status.READY) {
                m.setStatus(activeWorker >= targetWorker ? "Healthy" : "Unhealthy");
            } else {
                m.setStatus(status.name());
            }
        }
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
     * Returns the workflow version.
     *
     * @return the workflow version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns a list of models.
     *
     * @return a list of models
     */
    public List<Model> getModels() {
        return models;
    }

    /** A class represents model information. */
    public static final class Model {

        private String modelName;
        private String modelUrl;
        private int batchSize;
        private int maxBatchDelay;
        private int maxIdleTime;
        private int queueSize;
        private int requestInQueue;
        private String status;
        private boolean loadedAtStartup;

        private List<Group> workerGroups = new ArrayList<>();

        /**
         * Returns the model name.
         *
         * @return the model name
         */
        public String getModelName() {
            return modelName;
        }

        /**
         * Sets the model name.
         *
         * @param modelName the model name
         */
        public void setModelName(String modelName) {
            this.modelName = modelName;
        }

        /**
         * Returns the model URL.
         *
         * @return the model URL
         */
        public String getModelUrl() {
            return modelUrl;
        }

        /**
         * Sets the model URL.
         *
         * @param modelUrl the model URL
         */
        public void setModelUrl(String modelUrl) {
            this.modelUrl = modelUrl;
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
         * Returns the job queue size.
         *
         * @return the job queue size
         */
        public int getQueueSize() {
            return queueSize;
        }

        /**
         * Sets the job queue size.
         *
         * @param queueSize the job queue size
         */
        public void setQueueSize(int queueSize) {
            this.queueSize = queueSize;
        }

        /**
         * Returns the number of request in the queue.
         *
         * @return the number of request in the queue
         */
        public int getRequestInQueue() {
            return requestInQueue;
        }

        /**
         * Sets the number of request in the queue.
         *
         * @param requestInQueue the number of request in the queue
         */
        public void setRequestInQueue(int requestInQueue) {
            this.requestInQueue = requestInQueue;
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
         * Returns all workerPools of the workflow.
         *
         * @return all workerPools of the workflow
         */
        public List<Group> getWorkGroups() {
            return workerGroups;
        }

        void addGroup(Group group) {
            workerGroups.add(group);
        }
    }

    /** A class represents worker group. */
    public static final class Group {

        private Device device;
        private int minWorkers;
        private int maxWorkers;

        private List<Worker> workers;

        /**
         * Constructs a new instance of {@code Group}.
         *
         * @param device the device
         * @param minWorkers the minimum number of workers
         * @param maxWorkers the maximum number of workers
         */
        public Group(Device device, int minWorkers, int maxWorkers) {
            this.device = device;
            this.minWorkers = minWorkers;
            this.maxWorkers = maxWorkers;
            workers = new ArrayList<>();
        }

        /**
         * Returns the worker device.
         *
         * @return the worker device
         */
        public Device getDevice() {
            return device;
        }

        /**
         * Returns the minimum number of workers.
         *
         * @return the minimum number of workers
         */
        public int getMinWorkers() {
            return minWorkers;
        }

        /**
         * Returns the maximum number of workers.
         *
         * @return the maximum number of workers
         */
        public int getMaxWorkers() {
            return maxWorkers;
        }

        /**
         * Adds worker to the worker list.
         *
         * @param id the worker's ID
         * @param startTime the worker's start time
         * @param isRunning {@code true} if worker is running
         */
        public void addWorker(int id, long startTime, boolean isRunning) {
            Worker worker = new Worker();
            worker.setId(id);
            worker.setStartTime(new Date(startTime));
            worker.setStatus(isRunning ? "READY" : "UNLOADING");
            workers.add(worker);
        }

        /**
         * Returns a list of workers.
         *
         * @return a list of workers
         */
        public List<Worker> getWorkers() {
            return workers;
        }
    }

    /** A class that holds workers information. */
    public static final class Worker {

        private int id;
        private Date startTime;
        private String status;

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
    }
}
