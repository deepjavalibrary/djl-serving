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

import ai.djl.serving.workflow.WorkflowExpression.Item;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/** A simple {@link CacheEngine} that uses working memory. */
public class MemoryCacheEngine implements CacheEngine {

    Map<Integer, Item> cache;

    /**
     * Constructs a new {@link MemoryCacheEngine}.
     *
     * @param capacity the number of elements to cache
     */
    public MemoryCacheEngine(int capacity) {
        // Simple LRU cache based on https://stackoverflow.com/a/224886/3483497
        cache =
                new LinkedHashMap<>(capacity + 1, .75f, true) {
                    /** {@inheritDoc} */
                    @Override
                    public boolean removeEldestEntry(Entry<Integer, Item> eldest) {
                        return size() > capacity;
                    }
                };
        cache = Collections.synchronizedMap(cache);
    }

    /** {@inheritDoc} */
    @Override
    public void put(int hash, Item data) {
        cache.put(hash, data);
    }

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Optional<Item>> get(int hash) {
        return CompletableFuture.completedFuture(Optional.ofNullable(cache.get(hash)));
    }

    /** {@inheritDoc} */
    @Override
    public void clear(int hash) {
        cache.remove(hash);
    }
}
