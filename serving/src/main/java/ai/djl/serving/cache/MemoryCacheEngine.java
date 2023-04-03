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
package ai.djl.serving.cache;

import ai.djl.modality.Output;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link CacheEngine} that stores elements in working memory.
 *
 * <p>Note that this is not suitable if you expect the cache to work with a horizontally scaled
 * system.
 */
public class MemoryCacheEngine implements CacheEngine {

    private Map<String, Output> cache;
    private boolean multiTenant;

    /** Constructs a {@link MemoryCacheEngine}. */
    public MemoryCacheEngine() {
        cache = new ConcurrentHashMap<>();
    }

    /**
     * Constructs a {@link MemoryCacheEngine}.
     *
     * @param multiTenant whether to combine entries from multiple users
     */
    public MemoryCacheEngine(boolean multiTenant) {
        this();
        this.multiTenant = multiTenant;
    }

    /**
     * Constructs an LRU {@link MemoryCacheEngine} with limited capacity.
     *
     * @param multiTenant whether to combine entries from multiple users
     * @param capacity the maximum number of elements to store
     */
    public MemoryCacheEngine(boolean multiTenant, int capacity) {
        // Simple LRU cache based on https://stackoverflow.com/a/224886/3483497
        this.multiTenant = multiTenant;
        cache =
                new LinkedHashMap<>(capacity + 1, .75f, true) {
                    /** {@inheritDoc} */
                    @Override
                    public boolean removeEldestEntry(Map.Entry<String, Output> eldest) {
                        return size() > capacity;
                    }
                };
        cache = Collections.synchronizedMap(cache);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isMultiTenant() {
        return multiTenant;
    }

    /** {@inheritDoc} */
    @Override
    public void put(String key, Output output) {
        cache.put(key, output);
    }

    /** {@inheritDoc} */
    @Override
    public Output get(String key) {
        return cache.get(key);
    }

    /** {@inheritDoc} */
    @Override
    public void remove(String key) {
        cache.remove(key);
    }
}
