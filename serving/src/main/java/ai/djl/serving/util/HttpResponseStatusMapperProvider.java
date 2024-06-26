/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.util;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Supplier;

/**
 * Utility class that provides the HttpResponseStatusMapper to use in the InferenceRequestHandler.
 */
public final class HttpResponseStatusMapperProvider {

    private static final Map<String, Supplier<HttpResponseStatusMapper>> MAPPERS =
            new ConcurrentHashMap<>();

    private HttpResponseStatusMapperProvider() {}

    /**
     * Register a {@code HttpResponseStatusMapper} that can then be retrieved and used by the
     * InferenceRequestHandler.
     *
     * @param name the name of the mapper.
     * @param mapper the supplier used to construct an instance of the mapper.
     */
    public static void registerMapper(String name, Supplier<HttpResponseStatusMapper> mapper) {
        MAPPERS.put(name, mapper);
    }

    /**
     * Get the {@code HttpResponseStatusMapper} registered with the name.
     *
     * @param name the name the mapper was registered with.
     * @return the mapper to use in the InferenceRequestHandler.
     */
    public static HttpResponseStatusMapper getMapper(String name) {
        if (!MAPPERS.containsKey(name)) {
            return new DefaultHttpResponseStatusMapper();
        }
        return MAPPERS.get(name).get();
    }
}
