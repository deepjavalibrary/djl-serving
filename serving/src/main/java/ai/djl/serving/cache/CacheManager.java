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

/** A class that manages response cache. */
public final class CacheManager {

    private static CacheEngine instance = new MemoryCacheEngine();

    private CacheManager() {}

    /**
     * Returns the registered {@code CacheEngine} instance.
     *
     * @return the registered {@code CacheEngine} instance
     */
    public static CacheEngine getInstance() {
        return instance;
    }

    /**
     * Sets the {@code CacheEngine} instance.
     *
     * @param instance the {@code CacheEngine} instance
     */
    public static void setCacheManager(CacheEngine instance) {
        CacheManager.instance = instance;
    }
}
