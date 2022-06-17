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
package ai.djl.serving.wlm.util;

import ai.djl.serving.wlm.Job;

import java.util.concurrent.CompletableFuture;

/** A {@link Job} containing metadata from the {@link ai.djl.serving.wlm.WorkLoadManager}. */
public final class WorkerJob<I, O> {

    private final Job<I, O> job;
    private final CompletableFuture<O> future;

    /**
     * Constructs a new {@link WorkerJob}.
     *
     * @param job the job to execute
     * @param future the future containing the job response
     */
    public WorkerJob(Job<I, O> job, CompletableFuture<O> future) {
        this.job = job;
        this.future = future;
    }

    /**
     * Returns the {@link Job}.
     *
     * @return the {@link Job}
     */
    public Job<I, O> getJob() {
        return job;
    }

    /**
     * Returns the future for the job.
     *
     * @return the future for the job
     */
    public CompletableFuture<O> getFuture() {
        return future;
    }
}
