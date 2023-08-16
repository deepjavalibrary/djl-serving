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
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.serving.workflow.WorkflowExpression.Item.ItemType;
import ai.djl.serving.workflow.function.EnsembleMerge;
import ai.djl.serving.workflow.function.FunctionsApply;
import ai.djl.serving.workflow.function.IdentityWF;
import ai.djl.serving.workflow.function.WlmWorkflowFunction;
import ai.djl.serving.workflow.function.WorkflowFunction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collection;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/** A flow of executing {@link ai.djl.Model}s and custom functions. */
public class Workflow {

    private static final Logger logger = LoggerFactory.getLogger(Workflow.class);

    public static final String IN = "in";
    public static final String OUT = "out";

    private static final Map<String, Supplier<WorkflowFunction>> BUILT_INS =
            new ConcurrentHashMap<>();

    static {
        BUILT_INS.put(IdentityWF.NAME, IdentityWF::new);
        BUILT_INS.put(EnsembleMerge.NAME, EnsembleMerge::new);
        BUILT_INS.put(FunctionsApply.NAME, FunctionsApply::new);
    }

    String name;
    String version;
    Map<String, WorkerPoolConfig<Input, Output>> wpcs;
    Map<String, WorkflowExpression> expressions;
    Map<String, WorkflowFunction> funcs;
    Map<String, Map<String, Object>> configs;

    /**
     * Constructs a workflow containing only a single workerPoolConfig.
     *
     * @param wpc the workerPoolConfig for the workflow
     */
    public Workflow(WorkerPoolConfig<Input, Output> wpc) {
        String modelName = "model";
        this.name = wpc.getId();
        this.version = wpc.getVersion();
        wpcs = Collections.singletonMap(modelName, wpc);
        expressions =
                Collections.singletonMap(
                        OUT, new WorkflowExpression(new Item(modelName), new Item(IN)));
        funcs = Collections.emptyMap();
        configs = Collections.emptyMap();
    }

    /**
     * Constructs a workflow.
     *
     * @param name workflow name
     * @param version workflow version
     * @param wpcs a map of executableNames for a wpc (how it is referred to in the {@link
     *     WorkflowExpression}s to model
     * @param expressions a map of names to refer to an expression to the expression
     * @param configs the configuration objects
     * @param funcs the custom functions used in the workflow
     */
    public Workflow(
            String name,
            String version,
            Map<String, WorkerPoolConfig<Input, Output>> wpcs,
            Map<String, WorkflowExpression> expressions,
            Map<String, Map<String, Object>> configs,
            Map<String, WorkflowFunction> funcs) {
        this.name = name;
        this.version = version;
        this.wpcs = wpcs;
        this.expressions = expressions;
        this.funcs = funcs;
        this.configs = configs;
    }

    /**
     * Returns the {@link WorkerPoolConfig}s used in the workflow.
     *
     * @return the wpcs used in the workflow
     */
    public Collection<WorkerPoolConfig<Input, Output>> getWpcs() {
        return wpcs.values();
    }

    /**
     * Returns the wpc map in the workflow.
     *
     * @return the wpc map in the workflow
     */
    public Map<String, WorkerPoolConfig<Input, Output>> getWpcMap() {
        return wpcs;
    }

    /**
     * Executes a workflow with an input.
     *
     * @param wlm the wlm to run the workflow with
     * @param input the input
     * @return a future of the result of the execution
     */
    public CompletableFuture<Output> execute(WorkLoadManager wlm, Input input) {
        logger.trace("Beginning execution of workflow: {}", name);
        WorkflowExecutor ex = new WorkflowExecutor(wlm, input);
        return ex.execute(OUT)
                .thenApply(
                        i -> {
                            logger.trace("Ending execution of workflow: {}", name);
                            if (i.getItemType() != ItemType.INPUT) {
                                throw new IllegalArgumentException(
                                        "The workflow did not return an output. Instead it returned"
                                                + " an "
                                                + i.getItemType());
                            }
                            return (Output) i.getInput();
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

    /** Stops the workflow and unloads all the wpcs in the workflow. */
    public void stop() {
        for (WorkerPoolConfig<Input, Output> wpc : getWpcs()) {
            wpc.close();
        }
    }

    /** An executor is a session for a running {@link Workflow}. */
    public final class WorkflowExecutor {
        private WorkLoadManager wlm;
        private Map<String, CompletableFuture<Item>> vars;
        private Set<String> targetStack;

        private WorkflowExecutor(WorkLoadManager wlm, Input input) {
            this.wlm = wlm;

            // Construct variable map to contain each expression and the input
            vars = new ConcurrentHashMap<>(expressions.size() + 1);
            vars.put(IN, CompletableFuture.completedFuture(new Item(input)));

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
        public CompletableFuture<Item> execute(String target) {
            if (vars.containsKey(target)) {
                return vars.get(target);
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

            CompletableFuture<Item> result = executeExpression(expr);
            vars.put(target, result);
            return result.whenComplete(
                    (o, e) -> {
                        if (e != null) {
                            throw new WorkflowExecutionException(
                                    "Failed to compute workflow target: " + target, e);
                        }

                        targetStack.remove(target);
                        logger.trace("Workflow computed target {} with value:\n{}", target, o);
                    });
        }

        /**
         * Computes the result of a {@link WorkflowExpression}.
         *
         * @param expr the expression to compute
         * @return the computed value
         */
        public CompletableFuture<Item> executeExpression(WorkflowExpression expr) {
            WorkflowFunction workflowFunction = getExecutable(expr.getExecutableName());
            List<WorkflowArgument> args =
                    expr.getExecutableArgs().stream()
                            .map(arg -> new WorkflowArgument(this, arg))
                            .collect(Collectors.toList());
            return workflowFunction.run(this, args);
        }

        /**
         * Returns the executable (model, function, or built-in) with a given name.
         *
         * @param arg the workflow argument containing the name
         * @return the function to execute the found executable
         */
        public WorkflowFunction getExecutable(WorkflowArgument arg) {
            return getExecutable(arg.getItem().getString());
        }

        /**
         * Returns the executable (model, function, or built-in) with a given name.
         *
         * @param name the executable name
         * @return the function to execute the found executable
         */
        public WorkflowFunction getExecutable(String name) {
            WorkerPoolConfig<Input, Output> wpc = wpcs.get(name);
            if (wpc != null) {
                return new WlmWorkflowFunction(wpc);
            }

            if (funcs.containsKey(name)) {
                return funcs.get(name);
            }

            if (BUILT_INS.containsKey(name)) {
                // Built-in WorkflowFunctions should be one for each workflow
                WorkflowFunction f = BUILT_INS.get(name).get();
                funcs.put(name, f);
                return f;
            }

            throw new IllegalArgumentException("Could not find find model or function: " + name);
        }

        /**
         * Returns the configuration with the given name.
         *
         * @param name the configuration name
         * @return the configuration
         */
        public Map<String, Object> getConfig(WorkflowArgument name) {
            return getConfig(name.getItem().getString());
        }

        /**
         * Returns the configuration with the given name.
         *
         * @param name the configuration name
         * @return the configuration
         */
        public Map<String, Object> getConfig(String name) {
            return configs.get(name);
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
        public CompletableFuture<Item> evaluate() {
            switch (item.getItemType()) {
                case STRING:
                    return executor.execute(item.getString());
                case EXPRESSION:
                    return executor.executeExpression(item.getExpression());
                case INPUT:
                    return CompletableFuture.completedFuture(item);
                default:
                    throw new IllegalStateException(
                            "Found unexpected item type in workflow evaluate");
            }
        }
    }
}
