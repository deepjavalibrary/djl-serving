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

import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.serving.workflow.Workflow.WorkflowArgument;
import ai.djl.serving.workflow.Workflow.WorkflowExecutor;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.translate.Ensembleable;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Workflow function "ensemble" accepts a list of {@link Ensembleable} outputs and merges them using
 * {@link Ensembleable#ensemble(List)}.
 */
public class EnsembleMerge extends WorkflowFunction {

    public static final String NAME = "ensemble";

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public CompletableFuture<Item> run(WorkflowExecutor executor, List<WorkflowArgument> args) {
        if (args.size() != 1) {
            throw new IllegalArgumentException(
                    "Expected one arguments, the list of items to ensemble");
        }

        return CompletableFuture.supplyAsync(
                () -> {
                    List<Ensembleable> outputs =
                            args.get(0).evaluate().join().getList().stream()
                                    .map(i -> (Ensembleable<?>) i.getInput().get(0))
                                    .collect(Collectors.toList());

                    Ensembleable<?> ensembled = Ensembleable.ensemble(outputs);
                    Output output = new Output();
                    output.add((BytesSupplier) ensembled);
                    return new Item(output);
                });
    }
}
