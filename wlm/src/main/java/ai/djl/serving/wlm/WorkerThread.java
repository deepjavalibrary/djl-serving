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
import ai.djl.serving.wlm.WorkerPoolConfig.ThreadConfig;
import ai.djl.serving.wlm.util.AutoIncIdGenerator;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.serving.wlm.util.WorkerJob;
import ai.djl.translate.TranslateException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.time.Duration;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/** The {@link WorkerThread} is the worker managed by the {@link WorkLoadManager}. */
public final class WorkerThread<I, O> implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(WorkerThread.class);
    private static final AutoIncIdGenerator ID_GEN = new AutoIncIdGenerator("WT-");

    private ThreadConfig<I, O> threadConfig;
    private AtomicBoolean running = new AtomicBoolean(true);

    private LinkedBlockingDeque<WorkerJob<I, O>> configJobs;
    private BatchAggregator<I, O> aggregator;
    private Device device;
    private AtomicReference<Thread> currentThread = new AtomicReference<>();
    private WorkerState state;
    private String workerId;
    private long startTime;
    private boolean fixPoolThread;
    private long stateChangeTime;

    /**
     * Builds a workerThread with this builder.
     *
     * @param builder build a new worker thread using this builder.
     */
    private WorkerThread(Builder<I, O> builder) {
        this.aggregator = builder.aggregator;
        this.workerId = ID_GEN.generate();
        this.startTime = System.currentTimeMillis();
        this.fixPoolThread = builder.fixPoolThread;
        this.device = builder.device;
        threadConfig = builder.workerPoolConfig.newThread(device);
        configJobs = new LinkedBlockingDeque<>();

        logger.info(
                "Starting worker thread {} for model {} on device {}",
                workerId,
                builder.workerPoolConfig,
                device);
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        Thread thread = Thread.currentThread();
        thread.setName(workerId);
        currentThread.set(thread);
        this.state = WorkerState.WORKER_STARTED;
        List<Job<I, O>> req = null;
        String errorMessage = "Worker shutting down";
        try {
            runAllConfigJobs(); // Run initial config jobs
            while (isRunning() && !aggregator.isFinished()) {
                req = aggregator.getRequest();
                if (req != null && !req.isEmpty()) {
                    state = WorkerState.WORKER_BUSY;
                    runAllConfigJobs(); // Run new config jobs
                    try {
                        runJobs(req);
                        aggregator.sendResponse();
                    } catch (TranslateException e) {
                        logger.warn(workerId + ": Failed to predict", e);
                        aggregator.sendError(e);
                    } finally {
                        state = WorkerState.WORKER_STARTED;
                    }
                }
                req = null;
            }
        } catch (InterruptedException e) {
            logger.debug("Shutting down worker thread {} .. Scaling down.", workerId);
        } catch (Throwable t) {
            logger.error("{}: Server error", workerId, t);
            errorMessage = t.getMessage();
        } finally {
            logger.debug(
                    "Shutting down the worker thread {} .. {}",
                    workerId,
                    currentThread.get().getName());
            currentThread.set(null);
            shutdown(WorkerState.WORKER_STOPPED);
            if (req != null) {
                Exception e = new WlmException(errorMessage);
                aggregator.sendError(e);
            }
        }
    }

    private void runAllConfigJobs() throws TranslateException {
        while (!threadConfig.getConfigJobs().isEmpty()) {
            // Run base worker pool configurations if present
            runConfigJob(threadConfig.getConfigJobs().pop());
        }
        while (!configJobs.isEmpty()) {
            // Run thread config jobs if present
            runConfigJob(configJobs.pop().getJob());
        }
    }

    private O runConfigJob(Job<I, O> configJob) throws TranslateException {
        runJobs(Collections.singletonList(configJob));
        return configJob.getOutput();
    }

    private void runJobs(List<Job<I, O>> input) throws TranslateException {
        Map<Optional<JobFunction<I, O>>, List<Job<I, O>>> jobs =
                input.stream().collect(Collectors.groupingBy(Job::getRunner));
        for (Map.Entry<Optional<JobFunction<I, O>>, List<Job<I, O>>> fjob : jobs.entrySet()) {
            if (fjob.getKey().isPresent()) {
                Job.runAll(fjob.getValue(), fjob.getKey().get());
            } else {
                threadConfig.run(fjob.getValue());
            }
        }
    }

    /**
     * Returns the worker thread ID.
     *
     * @return the worker thread ID
     */
    public String getWorkerId() {
        return workerId;
    }

    /**
     * Returns the worker thread ID (number without prefix).
     *
     * @return the worker thread ID (number without prefix)
     */
    public int getWorkerIdNum() {
        return ID_GEN.stripPrefix(workerId);
    }

    /**
     * Returns the {@link WorkerPoolConfig}'s {@link ThreadConfig} for this thread.
     *
     * @return the {@link WorkerPoolConfig}'s {@link ThreadConfig} for this thread
     */
    public ThreadConfig<I, O> getThreadType() {
        return threadConfig;
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
        logger.info("Shutting down temporary worker {}", workerId);
        threadConfig.close();
    }

    /**
     * Adds a configuration job to this thread.
     *
     * @param wj the configuration job to add
     */
    public void addConfigJob(WorkerJob<I, O> wj) {
        configJobs.add(wj);
    }

    void setState(WorkerState newState) {
        logger.debug("Worker thread {} has state change: {} -> {}", workerId, state, newState);
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
     * @param <I> the workerPoolConfig input class
     * @param <O> the workerPoolConfig output class
     * @param wpc the {@link WorkerPoolConfig} the thread will be responsible for
     * @return a new builder
     */
    public static <I, O> Builder<I, O> builder(WorkerPoolConfig<I, O> wpc) {
        return new Builder<>(wpc);
    }

    /** A Builder to construct a {@code WorkerThread}. */
    public static final class Builder<I, O> {

        private WorkerPoolConfig<I, O> workerPoolConfig;
        private Device device;
        private BatchAggregator<I, O> aggregator;
        private LinkedBlockingDeque<WorkerJob<I, O>> jobQueue;
        private boolean fixPoolThread;

        Builder(WorkerPoolConfig<I, O> wpc) {
            this.workerPoolConfig = wpc;
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
                aggregator = new PermanentBatchAggregator<>(workerPoolConfig, jobQueue);
            } else {
                aggregator = new TemporaryBatchAggregator<>(workerPoolConfig, jobQueue);
            }
            return new WorkerThread<>(this);
        }
    }
}
