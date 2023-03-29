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

import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;

/** A class that manages response cache. */
public class CacheManager {

    private static CacheManager instance = new CacheManager();

    private Map<String, Output> cache = new ConcurrentHashMap<>();

    protected CacheManager() {}

    /**
     * Returns the registered {@code CacheManager} instance.
     *
     * @return the registered {@code CacheManager} instance
     */
    public static CacheManager getInstance() {
        return instance;
    }

    /**
     * Sets the {@code CacheManager} instance.
     *
     * @param instance the {@code CacheManager} instance
     */
    public static void setCacheManager(CacheManager instance) {
        CacheManager.instance = instance;
    }

    /**
     * Adds the {@code Output} to cache and return the cache key.
     *
     * @param output the {@code Output} to be added in cache
     * @return the cache key
     */
    public String put(Output output) {
        String key = UUID.randomUUID().toString();
        cache.put(key, output);
        return key;
    }

    /**
     * Returns the cached {@code Output} object witch the specified key.
     *
     * @param key the cache key
     * @return the cached {@code Output} object
     */
    public Output get(String key) {
        return cache.get(key);
    }

    /**
     * Removes the cache from the key.
     *
     * @param key the cache key
     */
    public void remove(String key) {
        cache.remove(key);
    }
}
