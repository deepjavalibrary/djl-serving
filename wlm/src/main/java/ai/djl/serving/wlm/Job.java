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
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;

import java.util.List;
import java.util.Optional;
import java.util.stream.Collectors;

/** A class represents an inference job. */
public class Job<I, O> {

    private WorkerPoolConfig<I, O> workerPoolConfig;
    private I input;
    private O output;
    private long begin;
    private JobFunction<I, O> runner;

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
     * Constructs a new {@code Job} instance.
     *
     * @param wpc the model to run the job
     * @param input the input data
     * @param runner the function to run on worker
     */
    public Job(WorkerPoolConfig<I, O> wpc, I input, JobFunction<I, O> runner) {
        this(wpc, input);
        this.runner = runner;
    }

    /**
     * Runs a {@link JobFunction} on a batch of jobs and sets the result in their output.
     *
     * @param jobs the jobs to run and update
     * @param f the function to run
     * @param <I> the input type
     * @param <O> the output type
     * @throws TranslateException if the jobs fail to run
     */
    public static <I, O> void runAll(List<Job<I, O>> jobs, JobFunction<I, O> f)
            throws TranslateException {
        List<O> out = f.apply(jobs.stream().map(Job::getInput).collect(Collectors.toList()));
        if (out != null) {
            for (int i = 0; i < out.size(); i++) {
                jobs.get(i).setOutput(out.get(i));
            }
        }
    }

    /**
     * Sets a {@link Job} output to a failure.
     *
     * @param job the job to set the output on
     * @param code the failure code
     * @param message the failure message
     */
    public static void setFailOutput(Job<Input, Output> job, int code, String message) {
        Output output = new Output();
        output.setCode(code);
        output.setMessage(message);
        job.setOutput(output);
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
     * Returns the output data.
     *
     * @return the output data
     */
    public O getOutput() {
        return output;
    }

    /**
     * Sets the output of the job.
     *
     * @param output the job output
     */
    public void setOutput(O output) {
        this.output = output;
    }

    /**
     * Returns the wait time of this job.
     *
     * @return the wait time of this job in mirco seconds
     */
    public long getWaitingMicroSeconds() {
        return (System.nanoTime() - begin) / 1000;
    }

    /**
     * Returns the task to run for the job.
     *
     * @return the task to run for the job
     */
    public Optional<JobFunction<I, O>> getRunner() {
        return Optional.ofNullable(runner);
    }
}
