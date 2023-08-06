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
import ai.djl.metric.Metrics;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.wlm.util.WorkerJob;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

/** The {@link WorkerThread} is the worker managed by the {@link WorkLoadManager}. */
public final class WorkerThread<I, O> implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private String workerName;
    private String modelName;
    private Predictor<I, O> predictor;

    private AtomicBoolean running = new AtomicBoolean(true);

    private BatchAggregator<I, O> aggregator;
    private Device device;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private WorkerState state;
    private int workerId;
    private long startTime;
    private boolean fixPoolThread;
    private boolean logModelMetric;
    private int metricsAggregation;
    private long stateChangeTime;

    /**
     * Builds a workerThread with this builder.
     *
     * @param builder build a new worker thread using this builder.
     */
    private WorkerThread(Builder<I, O> builder) {
        this.workerName = buildWorkerName(builder.model);
        this.aggregator = builder.aggregator;
        this.workerId = new WorkerIdGenerator().generate();
        this.startTime = System.currentTimeMillis();
        this.fixPoolThread = builder.fixPoolThread;
        this.device = builder.device;
        ZooModel<I, O> model = builder.model.getModel(device);

        predictor = model.newPredictor();
        modelName = builder.model.getId();
        logModelMetric = Boolean.parseBoolean(model.getProperty("log_model_metric"));
        metricsAggregation = Integer.parseInt(model.getProperty("metrics_aggregation", "1000"));
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        Thread thread = Thread.currentThread();
        thread.setName(workerName);
        currentThread.set(thread);
        this.state = WorkerState.WORKER_STARTED;
        List<I> req = null;
        String errorMessage = "Worker shutting down";
        try {
            if (logModelMetric) {
                Metrics metrics = new Metrics();
                metrics.setLimit(metricsAggregation);
                metrics.setOnLimit(
                        (m, s) -> MODEL_METRIC.info("{}-{}", modelName, m.percentile(s, 50)));
                predictor.setMetrics(metrics);
            }
            while (isRunning() && !aggregator.isFinished()) {
                req = aggregator.getRequest();
                if (req != null && !req.isEmpty()) {
                    state = WorkerState.WORKER_BUSY;
                    try {
                        List<O> reply = predictor.batchPredict(req);
                        aggregator.sendResponse(reply);
                    } catch (TranslateException e) {
                        logger.warn("Failed to predict", e);
                        aggregator.sendError(e);
                    } finally {
                        state = WorkerState.WORKER_STARTED;
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
        logger.info("shutdown temporary worker: {}", workerName);
        predictor.close();
    }

    private String buildWorkerName(ModelInfo<I, O> model) {
        String modelId = model.getId();
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
            stateChangeTime = System.currentTimeMillis();
        }
    }

    boolean isStale() {
        return (state == WorkerState.WORKER_STOPPED
                        || state == WorkerState.WORKER_ERROR
                        || state == WorkerState.WORKER_SCALED_DOWN)
                && System.currentTimeMillis() - stateChangeTime > Duration.ofMinutes(1).toMillis();
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
     * @param <I> the model input class
     * @param <O> the model output class
     * @param model the {@code ModelInfo} the thread will be responsible for
     * @return a new builder
     */
    public static <I, O> Builder<I, O> builder(ModelInfo<I, O> model) {
        return new Builder<>(model);
    }

    /** A Builder to construct a {@code WorkerThread}. */
    public static final class Builder<I, O> {

        private ModelInfo<I, O> model;
        private Device device;
        private BatchAggregator<I, O> aggregator;
        private LinkedBlockingDeque<WorkerJob<I, O>> jobQueue;
        private boolean fixPoolThread;

        Builder(ModelInfo<I, O> model) {
            this.model = model;
            this.fixPoolThread = true;
        }

        /**
         * RSets the device to run operations on.
         *
         * @param device the device to run operations on
         * @return self-reference to this builder
         */
        public Builder<I, O> setDevice(Device device) {
            this.device = device;
            return this;
        }

        /**
         * Sets the jobQueue used to poll for new jobs.
         *
         * @param jobQueue the jobQueue to set
         * @return self-reference to this builder.
         */
        public Builder<I, O> setJobQueue(LinkedBlockingDeque<WorkerJob<I, O>> jobQueue) {
            this.jobQueue = jobQueue;
            return this;
        }

        /**
         * Sets if the workerThread should be part of the fixed pool. Fixed Pool workers don't
         * terminate themself but are managed by WorkLoadManager min/max-worker scale functionality.
         *
         * @param fixPoolThread the fixPoolThread to set
         * @return self-reference to this builder.
         */
        public Builder<I, O> optFixPoolThread(boolean fixPoolThread) {
            this.fixPoolThread = fixPoolThread;
            return this;
        }

        /**
         * Builds the {@link WorkerThread} with the provided data.
         *
         * @return an {@link WorkerThread}
         */
        public WorkerThread<I, O> build() {
            if (device == null) {
                throw new IllegalArgumentException("Must set device for worker thread");
            }
            if (jobQueue == null) {
                throw new IllegalArgumentException("jobQueue has to be set.");
            }
            if (fixPoolThread) {
                aggregator = new PermanentBatchAggregator<>(model, jobQueue);
            } else {
                aggregator = new TemporaryBatchAggregator<>(model, jobQueue);
            }
            return new WorkerThread<>(this);
        }
    }
}
