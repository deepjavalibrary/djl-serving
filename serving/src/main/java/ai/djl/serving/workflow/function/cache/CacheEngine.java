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

import java.util.Optional;
import java.util.concurrent.CompletableFuture;

/** A cache that can be used as part of the {@link CacheWorkflowFunction}. */
public interface CacheEngine {

    /**
     * Puts an item in the cache.
     *
     * @param hash the item hash
     * @param data the data to store
     */
    void put(int hash, Item data);

    /**
     * Returns an item from the cache.
     *
     * @param hash the item hash
     * @return the item
     */
    CompletableFuture<Optional<Item>> get(int hash);

    /**
     * Removes an item from the cache.
     *
     * @param hash the item hash
     */
    void clear(int hash);
}
