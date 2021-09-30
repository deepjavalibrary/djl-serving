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

import ai.djl.modality.Input;

/** A class represents an inference job. */
public class Job {

    private ModelInfo modelInfo;
    private Input input;
    private long begin;
    private long scheduled;

    /**
     * Constructs a new {@code Job} instance.
     *
     * @param modelInfo the model to run the job
     * @param input the input data
     */
    public Job(ModelInfo modelInfo, Input input) {
        this.modelInfo = modelInfo;
        this.input = input;

        begin = System.currentTimeMillis();
        scheduled = begin;
    }

    /**
     * Returns the request id.
     *
     * @return the request id
     */
    public String getRequestId() {
        return input.getRequestId();
    }

    /**
     * Returns the model that associated with this job.
     *
     * @return the model that associated with this job
     */
    public ModelInfo getModel() {
        return modelInfo;
    }

    /**
     * Returns the input data.
     *
     * @return the input data
     */
    public Input getInput() {
        return input;
    }

    /**
     * Returns the job begin time.
     *
     * @return the job begin time
     */
    public long getBegin() {
        return begin;
    }

    /**
     * Returns the job scheduled time.
     *
     * @return the job scheduled time
     */
    public long getScheduled() {
        return scheduled;
    }

    /** Marks the job has been scheduled. */
    public void setScheduled() {
        scheduled = System.currentTimeMillis();
    }
}
