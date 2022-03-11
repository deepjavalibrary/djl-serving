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

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.serving.workflow.function.IdentityWF;
import ai.djl.serving.workflow.function.ModelWorkflowFunction;
import ai.djl.serving.workflow.function.WorkflowFunction;
import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** A flow of executing {@link ai.djl.Model}s and custom functions. */
public class Workflow implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(Workflow.class);

    public static final String IN = "in";
    public static final String OUT = "out";

    private static final Map<String, WorkflowFunction> BUILT_INS = new ConcurrentHashMap<>();

    static {
        BUILT_INS.put("id", new IdentityWF());
    }

    String name;
    String version;
    Map<String, ModelInfo> models;
    Map<String, WorkflowExpression> expressions;
    Map<String, WorkflowFunction> funcs;

    /**
     * Constructs a workflow containing only a single model.
     *
     * @param model the model for the workflow
     */
    public Workflow(ModelInfo model) {
        String modelName = "model";
        this.name = model.getModelId();
        this.version = model.getVersion();
        models = Collections.singletonMap(modelName, model);
        expressions =
                Collections.singletonMap(
                        OUT, new WorkflowExpression(new Item(modelName), new Item(IN)));
        funcs = Collections.emptyMap();
    }

    /**
     * Constructs a workflow.
     *
     * @param name workflow name
     * @param version workflow version
     * @param models a map of executableNames for a model (how it is referred to in the {@link
     *     WorkflowExpression}s to model
     * @param expressions a map of names to refer to an expression to the expression
     * @param funcs the custom functions used in the workflow
     */
    public Workflow(
            String name,
            String version,
            Map<String, ModelInfo> models,
            Map<String, WorkflowExpression> expressions,
            Map<String, WorkflowFunction> funcs) {
        this.name = name;
        this.version = version;
        this.models = models;
        this.expressions = expressions;
        this.funcs = funcs;
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
     * Load all the models in this workflow.
     *
     * @param device the device to load the models
     * @return a {@code CompletableFuture} instance
     */
    public CompletableFuture<Void> load(String device) {
        return CompletableFuture.supplyAsync(
                () -> {
                    try {
                        for (ModelInfo modelInfo : models.values()) {
                            String engine = modelInfo.getEngineName();
                            if (engine != null) {
                                DependencyManager dm = DependencyManager.getInstance();
                                dm.installEngine(engine);
                            }

                            modelInfo.load(device);
                        }
                    } catch (ModelException | IOException e) {
                        throw new CompletionException(e);
                    }
                    return null;
                });
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
        WorkflowExecutor ex = new WorkflowExecutor(wlm, input);
        return ex.execute(OUT)
                .thenApply(
                        i -> {
                            logger.debug("Ending execution of workflow");
                            return (Output) i;
                        });
    }

    /**
     * Returns the workflow name.
     *
     * @return the workflow name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the workflow version.
     *
     * @return the workflow version
     */
    public String getVersion() {
        return version;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof Workflow)) {
            return false;
        }
        Workflow p = (Workflow) o;
        return name.equals(p.getName()) && version.equals(p.getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(name, getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return name + ':' + version;
        }
        return name;
    }

    /** {@inheritDoc} * */
    @Override
    public void close() {
        for (ModelInfo m : getModels()) {
            m.close();
        }
    }

    /** An executor is a session for a running {@link Workflow}. */
    public final class WorkflowExecutor {
        private WorkLoadManager wlm;
        private Map<String, CompletableFuture<Input>> vars;
        private Set<String> targetStack;

        private WorkflowExecutor(WorkLoadManager wlm, Input input) {
            this.wlm = wlm;

            // Construct variable map to contain each expression and the input
            vars = new ConcurrentHashMap<>(expressions.size() + 1);
            vars.put(IN, CompletableFuture.completedFuture(input));

            targetStack = new HashSet<>();
        }

        /**
         * Returns the {@link WorkLoadManager} used by the {@link WorkflowExecutor}.
         *
         * @return the {@link WorkLoadManager} used by the {@link WorkflowExecutor}
         */
        public WorkLoadManager getWlm() {
            return wlm;
        }

        /**
         * Uses the execute to compute a local value or target.
         *
         * <p>These values can be found as the keys in the "workflow" object.
         *
         * @param target the target to compute
         * @return a future that contains the target value
         */
        public CompletableFuture<Input> execute(String target) {
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

            CompletableFuture<Input> result = executeExpression(expr);
            vars.put(target, result.thenApply(o -> o));
            return result.whenComplete(
                    (o, e) -> {
                        if (e != null) {
                            throw new WorkflowExecutionException(
                                    "Failed to compute workflow target: " + target, e);
                        }

                        targetStack.remove(target);
                        logger.debug(
                                "Workflow computed target "
                                        + target
                                        + " with value:\n"
                                        + o.toString());
                    });
        }

        /**
         * Computes the result of a {@link WorkflowExpression}.
         *
         * @param expr the expression to compute
         * @return the computed value
         */
        public CompletableFuture<Input> executeExpression(WorkflowExpression expr) {
            WorkflowFunction workflowFunction = getExecutable(expr.getExecutableName());
            List<WorkflowArgument> args =
                    expr.getExecutableArgs()
                            .stream()
                            .map(arg -> new WorkflowArgument(this, arg))
                            .collect(Collectors.toList());
            return workflowFunction.run(this, args);
        }

        /**
         * Returns the executable (model, function, or built-in) with a given name.
         *
         * @param name the executable name
         * @return the function to execute the found executable
         */
        public WorkflowFunction getExecutable(String name) {
            ModelInfo model = models.get(name);
            if (model != null) {
                return new ModelWorkflowFunction(model);
            }

            if (funcs.containsKey(name)) {
                return funcs.get(name);
            }

            if (BUILT_INS.containsKey(name)) {
                return BUILT_INS.get(name);
            }

            throw new IllegalArgumentException("Could not find find model or function: " + name);
        }
    }

    /** An argument that is passed to a {@link WorkflowFunction}. */
    public static class WorkflowArgument {
        private WorkflowExecutor executor;
        private Item item;

        /**
         * Constructs a {@link WorkflowArgument}.
         *
         * @param executor the executor associated with the argument
         * @param item the argument item
         */
        public WorkflowArgument(WorkflowExecutor executor, Item item) {
            this.executor = executor;
            this.item = item;
        }

        /**
         * Returns the item (either {@link String} or {@link WorkflowExpression}).
         *
         * @return the item (either {@link String} or {@link WorkflowExpression})
         */
        public Item getItem() {
            return item;
        }

        /**
         * Evaluates the argument as a target reference (if string) or function call (if
         * expression).
         *
         * @return the result of evaluating the argument
         */
        public CompletableFuture<Input> evaluate() {
            if (item.getString() != null) {
                return executor.execute(item.getString());
            } else {
                return executor.executeExpression(item.getExpression());
            }
        }
    }
}
