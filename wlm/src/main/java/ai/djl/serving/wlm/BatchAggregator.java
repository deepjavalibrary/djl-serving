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
import ai.djl.metric.MetricType;
import ai.djl.metric.Unit;
import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * abstract class for all BatchAggregators. A batch aggregator check working queue and combines
 * multiple job into one batch. batches of jobs are used cause optimisations in separate engines.
 *
 * @author erik.bamberg@web.de
 */
abstract class BatchAggregator<I, O> {

    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private Dimension dimension;
    protected int batchSize;
    protected long maxBatchDelayMicros;
    protected List<WorkerJob<I, O>> wjs;
    protected BlockingQueue<WorkerJob<I, O>> jobQueue;

    /**
     * Constructs a new {@code BbatchAggregator} instance.
     *
     * @param wpc the workerPoolConfig to use.
     * @param jobQueue the job queue for polling data from.
     */
    public BatchAggregator(WorkerPoolConfig<I, O> wpc, BlockingQueue<WorkerJob<I, O>> jobQueue) {
        this.dimension = new Dimension("Model", wpc.getId());
        this.batchSize = wpc.getBatchSize();
        this.maxBatchDelayMicros = wpc.getMaxBatchDelayMillis() * 1000L;
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
    public List<Job<I, O>> getRequest() throws InterruptedException {
        wjs = pollBatch();
        List<Job<I, O>> list = new ArrayList<>(wjs.size());
        for (WorkerJob<I, O> wj : wjs) {
            Job<I, O> job = wj.getJob();
            long queueTime = job.getWaitingMicroSeconds();
            Metric metric =
                    new Metric(
                            "QueueTime",
                            MetricType.HISTOGRAM,
                            queueTime,
                            Unit.MICROSECONDS,
                            dimension);
            MODEL_METRIC.info("{}", metric);
            list.add(job);
        }
        int size = list.size();
        if (size > 1) {
            MODEL_METRIC.info(
                    "{}", new Metric("DynamicBatchSize", size, Unit.COUNT_PER_ITEM, dimension));
        }
        return list;
    }

    /** Sends to response to all waiting clients. */
    public void sendResponse() {
        for (WorkerJob<I, O> wj : wjs) {
            wj.getFuture().complete(wj.getJob().getOutput());
            long latency = wj.getJob().getWaitingMicroSeconds();
            MODEL_METRIC.info(
                    "{}",
                    new Metric(
                            "RequestLatency",
                            MetricType.HISTOGRAM,
                            latency,
                            Unit.MICROSECONDS,
                            dimension));
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

    protected void drainTo(List<WorkerJob<I, O>> list, long maxDelay) throws InterruptedException {
        long begin = System.nanoTime();
        jobQueue.drainTo(list, batchSize - 1);
        int remain = batchSize - list.size();
        for (int i = 0; i < remain; ++i) {
            WorkerJob<I, O> wj = jobQueue.poll(maxDelay, TimeUnit.MICROSECONDS);
            if (wj == null || wj.getJob() == null) {
                break;
            }
            long end = System.nanoTime();
            maxDelay -= (end - begin) / 1000;
            begin = end;
            list.add(wj);
            if (maxDelay <= 0) {
                break;
            }
        }
    }
}
