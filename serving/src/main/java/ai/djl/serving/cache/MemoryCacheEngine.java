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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A {@link CacheEngine} that stores elements in working memory.
 *
 * <p>Note that this is not suitable if you expect the cache to work with a horizontally scaled
 * system.
 */
public class MemoryCacheEngine extends BaseCacheEngine {

    private Map<String, Item> cache;
    private boolean multiTenant;
    private boolean cleanOnAccess;

    /** Constructs a {@link MemoryCacheEngine}. */
    public MemoryCacheEngine() {
        this(false);
    }

    /**
     * Constructs a {@link MemoryCacheEngine}.
     *
     * @param multiTenant whether to combine entries from multiple users
     */
    public MemoryCacheEngine(boolean multiTenant) {
        cache = new ConcurrentHashMap<>();
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
                    public boolean removeEldestEntry(Map.Entry<String, Item> eldest) {
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
    public Output get(String key, int start, int limit) {
        Item item = cache.get(key);
        if (item == null) {
            return null;
        }

        Output output = new Output();
        List<byte[]> contents = new ArrayList<>();

        // Maybe add first contents from output
        if (start == 0) {
            output.setCode(item.output.getCode());
            output.setMessage(item.output.getMessage());
            output.setProperties(item.output.getProperties());
            if (item.output.getData() != null) {
                contents.add(item.output.getData().getAsBytes());
                limit--;
            }
        }

        // Add rest of contents from subsequent
        start++;
        int maxI = Math.min(start + limit, item.subsequent.size() + 1);
        maxI = maxI < 0 ? item.subsequent.size() + 1 : maxI; // Handle underflow on limit
        for (int i = start; i < maxI; i++) {
            contents.add(item.subsequent.get(i - 1));
        }
        if (!contents.isEmpty()) {
            output.add(joinBytes(contents));
        }

        // Handle if last of data or not
        boolean returnedLastItem = item.last && maxI == item.subsequent.size() + 1;
        if (!returnedLastItem && !output.getProperties().containsKey("x-next-token")) {
            output.addProperty("x-next-token", key + (maxI - 1));
            output.addProperty("X-Amzn-SageMaker-Custom-Attributes", "x-next-token=" + key);
        } else if (cleanOnAccess) { // Last item and should clean on access
            remove(key);
        }

        return output;
    }

    @Override
    protected void putSingle(String key, Output output, boolean last) {
        cache.put(key, new Item(output));
    }

    @Override
    protected void putStream(String key, Output output, byte[] buf, int index, boolean last) {
        cache.compute(
                key,
                (k, item) -> {
                    if (output != null || item == null) {
                        item = new Item(output);
                    }
                    if (buf != null) {
                        item.subsequent.add(buf);
                    }
                    item.last = last;
                    return item;
                });
    }

    /** {@inheritDoc} */
    @Override
    public void remove(String key) {
        cache.remove(key);
    }

    private static class Item {
        Output output;
        List<byte[]> subsequent;
        boolean last;

        public Item(Output output) {
            this.output = output;
            this.subsequent = new ArrayList<>();
        }
    }
}
