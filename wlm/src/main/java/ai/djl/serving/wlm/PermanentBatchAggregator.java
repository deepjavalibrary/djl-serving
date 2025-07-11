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
import java.util.concurrent.BlockingQueue;

/**
 * a batch aggregator that never terminates by itself. the surrounding thread has to be interrupted
 * by sending an interrupt signal.
 *
 * @author erik.bamberg@web.de
 */
public class PermanentBatchAggregator<I, O> extends BatchAggregator<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(PermanentBatchAggregator.class);

    /**
     * Constructs a {@code PermanentBatchAggregator} instance.
     *
     * @param wpc the workerPoolConfig to use.
     * @param jobQueue the job queue for polling data from.
     */
    public PermanentBatchAggregator(
            WorkerPoolConfig<I, O> wpc, BlockingQueue<WorkerJob<I, O>> jobQueue) {
        super(wpc, jobQueue);
    }

    /** {@inheritDoc} */
    @Override
    protected List<WorkerJob<I, O>> pollBatch() throws InterruptedException {
        List<WorkerJob<I, O>> list = new ArrayList<>(batchSize);
        WorkerJob<I, O> wj = jobQueue.take();
        list.add(wj);
        drainTo(list, maxBatchDelayMicros);
        logger.trace("sending jobs, size: {}", list.size());
        return list;
    }

    /** {@inheritDoc} */
    @Override
    public boolean isFinished() {
        return false;
    }
}
