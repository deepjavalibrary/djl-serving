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

import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/**
 * abstract class for all BatchAggregators. A batch aggregator check working queue and combines
 * multiple job into one batch. batches of jobs are used cause optimisations in separate engines.
 *
 * @author erik.bamberg@web.de
 */
abstract class BatchAggregator<I, O> {

    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");

    private Dimension dimension;
    protected int batchSize;
    protected int maxBatchDelay;
    protected List<WorkerJob<I, O>> wjs;
    protected LinkedBlockingDeque<WorkerJob<I, O>> jobQueue;

    /**
     * Constructs a new {@code BbatchAggregator} instance.
     *
     * @param model the model to use.
     * @param jobQueue the job queue for polling data from.
     */
    public BatchAggregator(ModelInfo<I, O> model, LinkedBlockingDeque<WorkerJob<I, O>> jobQueue) {
        this.dimension = new Dimension("Model", model.getModelId());
        this.batchSize = model.getBatchSize();
        this.maxBatchDelay = model.getMaxBatchDelay();
        this.jobQueue = jobQueue;
        wjs = new ArrayList<>();
    }

    /**
     * Poll the queue and return a list of Input Objects for the model.
     *
     * @return list of input objects to pass to the model.
     * @throws InterruptedException if thread gets interrupted while waiting for new data in the
     *     queue.
     */
    public List<I> getRequest() throws InterruptedException {
        wjs = pollBatch();
        List<I> list = new ArrayList<>(wjs.size());
        for (WorkerJob<I, O> wj : wjs) {
            Job<I, O> job = wj.getJob();
            long queueTime = job.getWaitingTime();
            SERVER_METRIC.info("{}", new Metric("QueueTime", queueTime, Unit.MICROSECONDS));
            list.add(job.getInput());
        }
        return list;
    }

    /**
     * Sends to response to all waiting clients.
     *
     * @param outputs list of model-outputs in same order as the input objects.
     */
    public void sendResponse(List<O> outputs) {
        if (wjs.size() != outputs.size()) {
            throw new IllegalStateException("Not all jobs get response.");
        }

        int i = 0;
        for (O output : outputs) {
            WorkerJob<I, O> wj = wjs.get(i++);
            wj.getFuture().complete(output);
            long latency = wj.getJob().getWaitingTime();
            Metric metric = new Metric("ModelLatency", latency, Unit.MICROSECONDS, dimension);
            SERVER_METRIC.info("{}", metric);
        }
        wjs.clear();
    }

    /**
     * Completes the job with an error.
     *
     * @param error the exception
     */
    public void sendError(Throwable error) {
        for (WorkerJob<I, O> wj : wjs) {
            wj.getFuture().completeExceptionally(error);
        }
        wjs.clear();
    }

    /**
     * Fills in the list with a batch of jobs.
     *
     * @return a list of jobs read by this batch interation.
     * @throws InterruptedException if interrupted
     */
    protected abstract List<WorkerJob<I, O>> pollBatch() throws InterruptedException;

    /**
     * Checks if this {@code BatchAggregator} and the thread can be shutdown or if this aggregator
     * waits for more data.
     *
     * @return true if we can shutdown the thread. for example when max idle time exceeded in
     *     temporary batch aggregator.
     */
    public abstract boolean isFinished();

    protected void drainTo(List<WorkerJob<I, O>> list, int maxDelay) throws InterruptedException {
        long begin = System.currentTimeMillis();
        jobQueue.drainTo(list, batchSize - 1);
        int remain = batchSize - list.size();
        for (int i = 0; i < remain; ++i) {
            WorkerJob<I, O> wj = jobQueue.poll(maxDelay, TimeUnit.MILLISECONDS);
            if (wj == null || wj.getJob() == null) {
                break;
            }
            long end = System.currentTimeMillis();
            maxDelay -= end - begin;
            begin = end;
            list.add(wj);
            if (maxDelay <= 0) {
                break;
            }
        }
    }
}
