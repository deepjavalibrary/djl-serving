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
package ai.djl.serving.http;

import ai.djl.serving.wlm.Adapter;

/** A class that holds information about adapter status. */
public class DescribeAdapterResponse {
    private String name;
    private String src;

    /**
     * Constructs a {@link DescribeAdapterResponse}.
     *
     * @param adapter the adapter to describe
     */
    public DescribeAdapterResponse(Adapter adapter) {
        this.name = adapter.getName();
        this.src = adapter.getSrc();
    }

    /**
     * Returns the adapter name.
     *
     * @return the adapter name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the adapter src.
     *
     * @return the adapter src
     */
    public String getSrc() {
        return src;
    }
}
