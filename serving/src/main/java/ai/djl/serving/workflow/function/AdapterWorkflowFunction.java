/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.serving.wlm.Adapter;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPool;
import ai.djl.serving.workflow.Workflow.WorkflowArgument;
import ai.djl.serving.workflow.Workflow.WorkflowExecutor;
import ai.djl.serving.workflow.WorkflowExpression;
import ai.djl.serving.workflow.WorkflowExpression.Item;

import java.net.URI;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Workflow function "adapter" applies an adapted model to an input.
 *
 * <p>To use this workflow function, you must pre-specify the adapted functions in the configs. In
 * the configs, create an object "adapters" with keys as reference names and values as objects. The
 * adapter reference objects should have three properties:
 *
 * <ul>
 *   <li>model - the model name
 *   <li>name - the adapter name
 *   <li>url - the adapter url
 * </ul>
 *
 * <p>To call this workflow function, it requires two arguments. The first is the adapter config
 * reference name (determining the model and adapter to use). The second argument is the input.
 *
 * <p>To see an example of this workflow function, see the <a
 * href="https://github.com/deepjavalibrary/djl-serving/tree/master/serving/src/test/resources/adapterWorkflows/w1/workflow.json">test
 * example</a>.
 */
public class AdapterWorkflowFunction extends WorkflowFunction {

    public static final String NAME = "adapter";

    private WorkLoadManager wlm;
    private Map<String, AdapterReference> adapters;

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings("unchecked")
    public void prepare(WorkLoadManager wlm, Map<String, Map<String, Object>> configs) {
        this.wlm = wlm;
        this.adapters = new ConcurrentHashMap<>();

        // Add adapters from configurations
        for (Map.Entry<String, Object> entry : configs.get("adapters").entrySet()) {
            String ref = entry.getKey();
            Map<String, Object> config = (Map<String, Object>) entry.getValue();
            String modelName = (String) config.get("model");
            String adapterName = (String) config.get("name");
            URI url = URI.create((String) config.get("url"));

            WorkerPool<?, ?> wp = wlm.getWorkerPoolById(modelName);
            Adapter adapter = Adapter.newInstance(wp, adapterName, url);
            adapters.put(ref, new AdapterReference(modelName, adapter));
        }

        // Register adapters
        for (AdapterReference adapter : adapters.values()) {
            adapter.adapter.register(wlm.getWorkerPoolById(adapter.modelName));
        }
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        for (AdapterReference adapter : adapters.values()) {
            WorkerPool<Input, Output> wp = wlm.getWorkerPoolById(adapter.modelName);
            if (wp != null) {
                Adapter.unregister(wp, adapter.adapter.getName());
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Item> run(WorkflowExecutor executor, List<WorkflowArgument> args) {
        if (args.size() != 2) {
            throw new IllegalArgumentException(
                    "The adapter workflow function should have two args, but has " + args.size());
        }
        String adapterReference = args.get(0).getItem().getString();

        if (!adapters.containsKey(adapterReference)) {
            throw new IllegalArgumentException(
                    "The adapter function was called with unknown adapter " + adapterReference);
        }
        AdapterReference adapter = adapters.get(adapterReference);

        return args.get(1)
                .evaluate()
                .thenComposeAsync(
                        evaluatedArg -> {
                            Input input = evaluatedArg.getInput();
                            input.add("adapter", adapter.adapter.getName());
                            return executor.executeExpression(
                                    new WorkflowExpression(
                                            new Item(adapter.modelName), new Item(input)));
                        });
    }

    private static final class AdapterReference {

        private String modelName;
        private Adapter adapter;

        private AdapterReference(String modelName, Adapter adapter) {
            this.modelName = modelName;
            this.adapter = adapter;
        }
    }
}
