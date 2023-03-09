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
package ai.djl.serving.workflow.function.cache;

import ai.djl.serving.workflow.Workflow.WorkflowArgument;
import ai.djl.serving.workflow.Workflow.WorkflowExecutor;
import ai.djl.serving.workflow.WorkflowExpression.Item;
import ai.djl.serving.workflow.function.WorkflowFunction;
import ai.djl.translate.ArgumentsUtil;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

/**
 * {@link WorkflowFunction} "cache" that caches the results of the computation.
 *
 * <p>It should be called with three arguments such as ["cache", "modelName", "configName"].
 *
 * <p>The following config options are available:
 *
 * <ul>
 *   <li>cache - the type of cache. Available options are "memory".
 *   <li>capacity - the number of elements to cache. Only used by some cache engines including
 *       "memory".
 *   <li>usageTimeout - number of minutes between creating a cache element and it being removed.
 * </ul>
 */
public class CacheWorkflowFunction extends WorkflowFunction {

    public static final String NAME = "cache";

    static final Logger logger = LoggerFactory.getLogger(CacheWorkflowFunction.class);

    private Map<String, CacheEngine> caches;

    /** Constructs a {@link CacheWorkflowFunction}. */
    public CacheWorkflowFunction() {
        caches = new ConcurrentHashMap<>();
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Item> run(WorkflowExecutor executor, List<WorkflowArgument> args) {
        if (args.size() != 3) {
            throw new IllegalArgumentException(
                    "The cache should have three args: model, input, settings but found "
                            + args.size());
        }

        String modelName = args.get(0).getItem().getString();
        WorkflowFunction model = executor.getExecutable(modelName);
        WorkflowArgument input = args.get(1);
        Map<String, Object> config = executor.getConfig(args.get(2));

        CacheEngine cache = getOrCreateCacheEngine(modelName, config);

        return input.evaluate()
                .thenComposeAsync(
                        processedInput -> {
                            int hash =
                                    Arrays.hashCode(
                                            processedInput.getInput().getData().getAsBytes());

                            // Try using cache
                            return cache.get(hash)
                                    .thenComposeAsync(
                                            lookup -> {
                                                if (lookup.isPresent()) {
                                                    return CompletableFuture.completedFuture(
                                                            lookup.get());
                                                }

                                                // Compute value through model
                                                CompletableFuture<Item> output =
                                                        model.run(
                                                                        executor,
                                                                        Collections.singletonList(
                                                                                input))
                                                                .whenComplete(
                                                                        (item, e) ->
                                                                                cache.put(
                                                                                        hash,
                                                                                        item));

                                                // Clear if using usageTimeout
                                                if (config.containsKey("usageTimeout")) {
                                                    int usageTimeout =
                                                            ArgumentsUtil.intValue(
                                                                    config, "usageTimeout");
                                                    output.whenCompleteAsync(
                                                            (res, e) -> cache.clear(hash),
                                                            CompletableFuture.delayedExecutor(
                                                                    usageTimeout,
                                                                    TimeUnit.MINUTES));
                                                }
                                                return output;
                                            });
                        });
    }

    private CacheEngine getOrCreateCacheEngine(String modelName, Map<String, Object> config) {
        if (caches.containsKey(modelName)) {
            return caches.get(modelName);
        }

        CacheEngine engine = constructCacheEngine(config);
        caches.put(modelName, engine);
        return engine;
    }

    private CacheEngine constructCacheEngine(Map<String, Object> config) {
        String cacheType = ArgumentsUtil.stringValue(config, "cache");
        switch (cacheType) {
            case "memory":
                int capacity = ArgumentsUtil.intValue(config, "capacity");
                if (capacity == 0) {
                    capacity = 128;
                    logger.info(
                            "The MemoryCacheEngine was not provided a capacity and is using the"
                                    + " default value: "
                                    + capacity);
                }
                return new MemoryCacheEngine(capacity);
            default:
                throw new IllegalArgumentException(
                        "Workflow provided invalid or missing cache type: " + cacheType);
        }
    }
}
