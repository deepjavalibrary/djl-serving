/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.workflow.function;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowExpression.Item;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * An internal {@link WorkflowFunction} that is used to execute a {@link WorkerPoolConfig}
 * (typically a model) through the {@link ai.djl.serving.wlm.WorkLoadManager} in the workflow.
 */
public class WlmWorkflowFunction extends WorkflowFunction {

    WorkerPoolConfig<Input, Output> workerPoolConfig;

    /**
     * Constructs a {@link WlmWorkflowFunction} with a given workerPoolConfig.
     *
     * @param wpc the workerPoolConfig to run
     */
    public WlmWorkflowFunction(WorkerPoolConfig<Input, Output> wpc) {
        this.workerPoolConfig = wpc;
    }

    /** {@inheritDoc} */
    @SuppressWarnings("unchecked")
    @Override
    public CompletableFuture<Item> run(
            Workflow.WorkflowExecutor executor, List<Workflow.WorkflowArgument> args) {
        if (args.size() != 1) {
            throw new IllegalArgumentException(
                    "The model or worker type "
                            + workerPoolConfig.getId()
                            + " should have one arg, but has "
                            + args.size());
        }

        return evaluateArgs(args)
                .thenCompose(
                        processedArgs ->
                                executor.getWlm()
                                        .runJob(
                                                new Job<>(
                                                        workerPoolConfig,
                                                        processedArgs.get(0).getInput()))
                                        .thenApply(Item::new));
    }
}
