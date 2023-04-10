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

import ai.djl.modality.Input;
import ai.djl.modality.Output;

import java.util.Arrays;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;

/**
 * A cache that can be used for streaming online caching, streaming pagination, or async
 * predictions.
 */
public interface CacheEngine {

    /**
     * Returns whether the cache should combine results from multiple users.
     *
     * @return whether the cache should combine results from multiple users
     */
    boolean isMultiTenant();

    /**
     * Creates a new key to store in the cache.
     *
     * @return the cache key
     */
    default String create() {
        return UUID.randomUUID().toString();
    }

    /**
     * Adds the {@code Output} to cache and return the cache key.
     *
     * @param input the {@code Input} that could be used to build a key
     * @return the cache key
     */
    default String create(Input input) {
        if (isMultiTenant() && input != null) {
            int hash = Arrays.hashCode(input.getData().getAsBytes());
            return String.valueOf(hash);
        }

        return create();
    }

    /**
     * Adds the {@code Output} to cache and return the cache key.
     *
     * @param key the cache key
     * @param output the {@code Output} to be added in cache
     * @return a {@code CompletableFuture} instance
     */
    CompletableFuture<Void> put(String key, Output output);

    /**
     * Adds the {@code Output} to cache and return the cache key.
     *
     * @param input the {@code Input} that could be used to build a key
     * @param output the {@code Output} to be added in cache
     * @return the cache key
     */
    default String put(Input input, Output output) {
        String key = create(input);
        put(key, output);
        return key;
    }

    /**
     * Returns the cached {@link Output} from the cache.
     *
     * @param key the cache key
     * @param limit the max number of items to return
     * @return the item
     */
    Output get(String key, int limit);

    /**
     * Removes the cache from the key.
     *
     * @param key the cache key
     */
    void remove(String key);
}
