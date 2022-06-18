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

import java.util.List;
import java.util.concurrent.CompletableFuture;

/** An identity function which passes back it's input. */
public class IdentityWF extends WorkflowFunction {

    @Override
    public CompletableFuture<Input> run(
            Workflow.WorkflowExecutor executor, List<Workflow.WorkflowArgument> args) {
        if (args.size() != 1) {
            throw new IllegalArgumentException("Expected one argument to id");
        }
        return evaluateArgs(args).thenApply(pa -> pa.get(0));
    }
}
