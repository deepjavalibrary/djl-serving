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

import ai.djl.serving.workflow.Workflow.WorkflowArgument;
import ai.djl.serving.workflow.Workflow.WorkflowExecutor;
import ai.djl.serving.workflow.WorkflowExpression;
import ai.djl.serving.workflow.WorkflowExpression.Item;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Workflow function "functionsApply" accepts a list of functions and an input and applies each
 * function to the input.
 *
 * <p>It returns a list of the results of applying each function to the input.
 */
public class FunctionsApply extends WorkflowFunction {

    public static final String NAME = "functionsApply";

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public CompletableFuture<Item> run(WorkflowExecutor executor, List<WorkflowArgument> args) {
        if (args.size() != 2) {
            throw new IllegalArgumentException(
                    "Expected two arguments: the list of functions to run and the input, but found "
                            + args.size());
        }
        List<Item> fns = args.get(0).getItem().getList();
        Item input = args.get(1).getItem();

        return CompletableFuture.supplyAsync(
                () -> {
                    // Get classifications
                    CompletableFuture<Item>[] futures =
                            fns.stream()
                                    .map(
                                            fn ->
                                                    executor.executeExpression(
                                                            new WorkflowExpression(fn, input)))
                                    .toArray(CompletableFuture[]::new);
                    CompletableFuture.allOf(futures);
                    List<Item> outputs =
                            Arrays.stream(futures).map(m -> m.join()).collect(Collectors.toList());

                    return new Item(new WorkflowExpression(outputs));
                });
    }
}
