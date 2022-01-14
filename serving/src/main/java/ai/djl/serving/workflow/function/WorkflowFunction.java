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
import ai.djl.serving.workflow.Workflow;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * A lambda function that can be run within a {@link Workflow}.
 *
 * @see #run(Workflow.WorkflowExecutor, List)
 */
public abstract class WorkflowFunction {

    /**
     * The lambda function that is run.
     *
     * @param executor an executor that can be used to run expressions or models
     * @param args the list of function arguments
     * @return a future containing the input
     */
    public abstract CompletableFuture<Input> run(
            Workflow.WorkflowExecutor executor, List<Workflow.WorkflowArgument> args);

    /**
     * A helper to evaluate all function arguments.
     *
     * @param args the arguments to evaluate
     * @return a future with the list of evaluated arguments
     */
    @SuppressWarnings("unchecked")
    protected CompletableFuture<List<Input>> evaluateArgs(List<Workflow.WorkflowArgument> args) {
        CompletableFuture<Input>[] processedArgs =
                args.stream()
                        .map(Workflow.WorkflowArgument::evaluate)
                        .toArray(CompletableFuture[]::new);

        return CompletableFuture.allOf(processedArgs)
                .thenApply(
                        v ->
                                Arrays.stream(processedArgs)
                                        .map(CompletableFuture::join)
                                        .collect(Collectors.toList()));
    }
}
