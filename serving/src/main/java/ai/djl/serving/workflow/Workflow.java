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
package ai.djl.serving.workflow;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A flow of executing {@link ai.djl.Model}s. */
public class Workflow implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Workflow.class);

    public static final String IN = "in";
    public static final String OUT = "out";

    Map<String, ModelInfo> models;
    Map<String, WorkflowExpression> expressions;

    /**
     * Constructs a workflow containing only a single model.
     *
     * @param model the model for the workflow
     */
    public Workflow(ModelInfo model) {
        String modelName = "model";
        models = Collections.singletonMap(modelName, model);
        expressions =
                Collections.singletonMap(
                        OUT, new WorkflowExpression(modelName, Collections.singletonList(IN)));
    }

    /**
     * Constructs a workflow.
     *
     * @param models a map of executableNames for a model (how it is referred to in the {@link
     *     WorkflowExpression}s to model
     * @param expressions a map of names to refer to an expression to the expression
     */
    public Workflow(Map<String, ModelInfo> models, Map<String, WorkflowExpression> expressions) {
        this.models = models;
        this.expressions = expressions;
    }

    /**
     * Returns the models used in the workflow.
     *
     * @return the models used in the workflow
     */
    public Collection<ModelInfo> getModels() {
        return models.values();
    }

    /**
     * Executes a workflow with an input.
     *
     * @param wlm the wlm to run the workflow with
     * @param input the input
     * @return a future of the result of the execution
     */
    public CompletableFuture<Output> execute(WorkLoadManager wlm, Input input) {
        logger.debug("Beginning execution of workflow");
        // Construct variable map to contain each expression and the input
        Map<String, CompletableFuture<Input>> vars =
                new ConcurrentHashMap<>(expressions.size() + 1);
        vars.put(IN, CompletableFuture.completedFuture(input));

        return execute(wlm, OUT, vars, new HashSet<>()).thenApply(i -> (Output) i);
    }

    @SuppressWarnings("unchecked")
    private CompletableFuture<Input> execute(
            WorkLoadManager wlm,
            String target,
            Map<String, CompletableFuture<Input>> vars,
            Set<String> targetStack) {
        if (vars.containsKey(target)) {
            return vars.get(target).thenApply(i -> i);
        }

        // Use targetStack, the set of targets in the "call stack" to detect cycles
        if (targetStack.contains(target)) {
            // If a target is executed but already in the stack, there must be a cycle
            throw new IllegalStateException(
                    "Your workflow contains a cycle with target: " + target);
        }
        targetStack.add(target);

        WorkflowExpression expr = expressions.get(target);
        if (expr == null) {
            throw new IllegalArgumentException(
                    "Expected to find variable but it is not defined: " + target);
        }

        ModelInfo model = models.get(expr.getExecutableName());
        if (model == null) {
            throw new IllegalArgumentException(
                    "Expected to find model but it is not defined: " + target);
        }

        CompletableFuture<Input>[] processedArgs =
                expr.getArgs()
                        .stream()
                        .map(arg -> execute(wlm, arg, vars, targetStack))
                        .toArray(CompletableFuture[]::new);

        if (processedArgs.length != 1) {
            throw new IllegalArgumentException(
                    "In the definition for "
                            + target
                            + ", the model "
                            + expr.getExecutableName()
                            + " should have one arg, but has "
                            + processedArgs.length);
        }

        return CompletableFuture.allOf(processedArgs)
                .thenApply(
                        v -> {
                            CompletableFuture<Output> result =
                                    wlm.runJob(new Job(model, processedArgs[0].join()));
                            vars.put(target, result.thenApply(o -> o));
                            result.thenAccept(
                                    r -> {
                                        targetStack.remove(target);
                                        logger.debug(
                                                "Workflow computed target "
                                                        + target
                                                        + " with value:\n"
                                                        + r.toString());
                                    });
                            return result.join();
                        });
    }

    /** {@inheritDoc} * */
    @Override
    public void close() {
        for (ModelInfo m : getModels()) {
            m.close();
        }
    }
}
