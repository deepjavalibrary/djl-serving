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
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.wlm.util.WorkerJob;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** The {@link WorkerThread} is the worker managed by the {@link WorkLoadManager}. */
public final class WorkerThread implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);

    private String workerName;
    private Predictor<Input, Output> predictor;

    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator aggregator;
    private Device device;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private WorkerState state;
    private int workerId;
    private long startTime;
    private boolean fixPoolThread;

    /**
     * Builds a workerThread with this builder.
     *
     * @param builder build a new worker thread using this builder.
     */
    private WorkerThread(Builder builder) {
        this.workerName = buildWorkerName(builder.model);
        this.aggregator = builder.aggregator;
        this.workerId = new WorkerIdGenerator().generate();
        this.startTime = System.currentTimeMillis();
        this.fixPoolThread = builder.fixPoolThread;
        this.device = builder.device;
        ZooModel<Input, Output> model = builder.model.getModel(device);

        predictor = model.newPredictor();
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        Thread thread = Thread.currentThread();
        thread.setName(workerName);
        currentThread.set(thread);
        this.state = WorkerState.WORKER_STARTED;
        List<Input> req = null;
        String errorMessage = "Worker shutting down";
        try {
            while (isRunning() && !aggregator.isFinished()) {
                req = aggregator.getRequest();
                if (req != null && !req.isEmpty()) {
                    try {
                        List<Output> reply = predictor.batchPredict(req);
                        aggregator.sendResponse(reply);
                    } catch (Exception e) {
                        logger.warn("Failed to predict", e);
                        aggregator.sendError(e);
                    }
                }
                req = null;
            }
        } catch (InterruptedException e) {
            logger.debug("Shutting down the thread .. Scaling down.");
        } catch (Throwable t) {
            logger.error("Server error", t);
            errorMessage = t.getMessage();
        } finally {
            logger.debug("Shutting down worker thread .. {}", currentThread.get().getName());
            currentThread.set(null);
            shutdown(WorkerState.WORKER_STOPPED);
            if (req != null) {
                Exception e = new WlmException(errorMessage);
                aggregator.sendError(e);
            }
        }
    }

    /**
     * Returns the worker thread ID.
     *
     * @return the worker thread ID
     */
    public int getWorkerId() {
        return workerId;
    }

    /**
     * Returns true if the worker thread is running.
     *
     * @return true if the worker thread is running
     */
    public boolean isRunning() {
        return running.get();
    }

    /**
     * Returns the device used by the thread.
     *
     * @return the device used by the thread
     */
    public Device getDevice() {
        return device;
    }

    /**
     * Returns the thread start time.
     *
     * @return the thread start time
     */
    public long getStartTime() {
        return startTime;
    }

    /**
     * Returns the worker state.
     *
     * @return the worker state
     */
    public WorkerState getState() {
        return state;
    }

    /**
     * Shuts down the worker thread.
     *
     * @param state the state to set the thread to
     */
    public void shutdown(WorkerState state) {
        running.set(false);
        setState(state);
        Thread thread = currentThread.getAndSet(null);
        if (thread != null) {
            thread.interrupt();
            Exception e = new WlmException("Worker shutting down");
            aggregator.sendError(e);
        }
        predictor.close();
    }

    private String buildWorkerName(ModelInfo model) {
        String modelId = model.getModelId();
        if (modelId.length() > 25) {
            modelId = modelId.substring(0, 25);
        }
        return "W-" + modelId + '-' + workerId;
    }

    void setState(WorkerState newState) {
        logger.debug("{} State change {} -> {}", workerName, state, newState);
        if (state != WorkerState.WORKER_SCALED_DOWN) {
            // Don't update the state if it was terminated on purpose.. Scaling in..
            this.state = newState;
        }
    }

    /**
     * check if this worker is instantiate is one of the fix threads of a pool. fix threads are not
     * automatically scales down, so they are candidate for down scaling when minWorker/maxWorker
     * size of a model changes.
     *
     * @return the fixPoolThread
     */
    public boolean isFixPoolThread() {
        return fixPoolThread;
    }

    /**
     * Creates a builder to build a {@code WorkerThread}.
     *
     * @return a new builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** A Builder to construct a {@code WorkerThread}. */
    public static class Builder {

        private ModelInfo model;
        private Device device;
        private BatchAggregator aggregator;
        private LinkedBlockingDeque<WorkerJob> jobQueue;
        private boolean fixPoolThread;

        Builder() {
            this.fixPoolThread = true;
        }

        /**
         * Returns self reference to this builder.
         *
         * @return self reference to this builder
         */
        protected Builder self() {
            return this;
        }

        protected void preBuildProcessing() {
            if (aggregator == null) {
                if (fixPoolThread) {
                    aggregator = new PermanentBatchAggregator(model, jobQueue);
                } else {
                    aggregator = new TemporaryBatchAggregator(model, jobQueue);
                }
            }
        }

        protected void validate() {
            if (device == null) {
                throw new IllegalArgumentException("Must set device for worker thread");
            }
            if (model == null) {
                throw new IllegalArgumentException("Must set model for worker thread");
            }
            if (jobQueue == null && aggregator == null) {
                throw new IllegalArgumentException(
                        "one of jobQueue or BatchAggregator have to be set.");
            }
        }

        /**
         * Builds the {@link WorkerThread} with the provided data.
         *
         * @return an {@link WorkerThread}
         */
        public WorkerThread build() {
            validate();
            preBuildProcessing();
            return new WorkerThread(this);
        }

        /**
         * Sets the {@code ModelInfo} the thread will be responsible for.
         *
         * @param model the model to set
         * @return self-reference to this builder.
         */
        public Builder setModel(ModelInfo model) {
            this.model = model;
            return self();
        }

        /**
         * RSets the device to run operations on.
         *
         * @param device the device to run operations on
         * @return self-reference to this builder
         */
        public Builder setDevice(Device device) {
            this.device = device;
            return self();
        }

        /**
         * Sets a {@code BatchAggregator} which overrides the instantiated default {@code
         * BatchAggregator}.
         *
         * @param aggregator the {@code BatchAggregator} to set
         * @return self-reference to this builder.
         */
        public Builder optAggregator(BatchAggregator aggregator) {
            this.aggregator = aggregator;
            return self();
        }

        /**
         * Sets the jobQueue used to poll for new jobs. The jobQueue is passed to the created
         * standard BatchAggregators if the Batch-Aggregator is not override using {@link
         * #optAggregator(BatchAggregator) optAggregator(BatchAggregator)}
         *
         * @param jobQueue the jobQueue to set
         * @return self-reference to this builder.
         */
        public Builder setJobQueue(LinkedBlockingDeque<WorkerJob> jobQueue) {
            this.jobQueue = jobQueue;
            return self();
        }

        /**
         * Sets if the workerThread should be part of the fixed pool. Fixed Pool workers don't
         * terminate themself but are managed by WorkLoadManager min/max-worker scale functionality.
         *
         * @param fixPoolThread the fixPoolThread to set
         * @return self-reference to this builder.
         */
        public Builder optFixPoolThread(boolean fixPoolThread) {
            this.fixPoolThread = fixPoolThread;
            return self();
        }
    }
}
