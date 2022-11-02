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

import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowExpression.Item;

import java.util.List;
import java.util.concurrent.CompletableFuture;

/** Workflow function "id" accepts a single argument and returns the result of evaluating it. */
public class IdentityWF extends WorkflowFunction {

    public static final String NAME = "id";

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Item> run(
            Workflow.WorkflowExecutor executor, List<Workflow.WorkflowArgument> args) {
        if (args.size() != 1) {
            throw new IllegalArgumentException("Expected one argument to id");
        }
        return evaluateArgs(args).thenApply(pa -> pa.get(0));
    }
}
