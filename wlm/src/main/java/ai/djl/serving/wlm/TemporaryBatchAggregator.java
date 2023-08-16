/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.serving.wlm.util.WorkerJob;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.TimeUnit;

/**
 * a batch aggregator that terminates after a maximum idle time.
 *
 * @author erik.bamberg@web.de
 */
public class TemporaryBatchAggregator<I, O> extends BatchAggregator<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(TemporaryBatchAggregator.class);

    private long idleSince;
    private long maxIdleSeconds;

    /**
     * a batch aggregator that terminates after a maximum idle time.
     *
     * @param wpc the workerPoolConfig to run for.
     * @param jobQueue reference to external job queue for polling.
     */
    public TemporaryBatchAggregator(
            WorkerPoolConfig<I, O> wpc, LinkedBlockingDeque<WorkerJob<I, O>> jobQueue) {
        super(wpc, jobQueue);
        this.idleSince = System.currentTimeMillis();
        this.maxIdleSeconds = wpc.getMaxIdleSeconds();
    }

    /** {@inheritDoc} */
    @Override
    protected List<WorkerJob<I, O>> pollBatch() throws InterruptedException {
        List<WorkerJob<I, O>> list = new ArrayList<>(batchSize);
        WorkerJob<I, O> wj = jobQueue.poll(maxIdleSeconds, TimeUnit.SECONDS);
        if (wj != null && wj.getJob() != null) {
            list.add(wj);
            drainTo(list, maxBatchDelayMicros);
            logger.trace("sending jobs, size: {}", list.size());
            idleSince = System.currentTimeMillis();
        }
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isFinished() {
        long idle = System.currentTimeMillis() - idleSince;
        logger.trace("Temporary batch aggregator idle time (max {}s): {}ms", maxIdleSeconds, idle);
        return idle > maxIdleSeconds * 1000;
    }
}
