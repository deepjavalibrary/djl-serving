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

    /**
     * Constructs a new {@code Job} instance.
     *
     * @param modelInfo the model to run the job
     * @param input the input data
     */
    public Job(ModelInfo modelInfo, Input input) {
        this.modelInfo = modelInfo;
        this.input = input;

        begin = System.nanoTime();
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
     * Returns the wait time of this job.
     *
     * @return the wait time of this job in mirco seconds
     */
    public long getWaitingTime() {
        return (System.nanoTime() - begin) / 1000;
    }
}
