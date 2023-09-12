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

/** A class represents an inference job. */
public class Job<I, O> {

    private WorkerPoolConfig<I, O> workerPoolConfig;
    private I input;
    private long begin;

    /**
     * Constructs a new {@code Job} instance.
     *
     * @param wpc the model to run the job
     * @param input the input data
     */
    public Job(WorkerPoolConfig<I, O> wpc, I input) {
        this.workerPoolConfig = wpc;
        this.input = input;

        begin = System.nanoTime();
    }

    /**
     * Returns the worker pool config that is associated with this job.
     *
     * @return the worker pool config that is associated with this job
     */
    public WorkerPoolConfig<I, O> getWpc() {
        return workerPoolConfig;
    }

    /**
     * Returns the input data.
     *
     * @return the input data
     */
    public I getInput() {
        return input;
    }

    /**
     * Returns the wait time of this job.
     *
     * @return the wait time of this job in mirco seconds
     */
    public long getWaitingMicroSeconds() {
        return (System.nanoTime() - begin) / 1000;
    }
}
