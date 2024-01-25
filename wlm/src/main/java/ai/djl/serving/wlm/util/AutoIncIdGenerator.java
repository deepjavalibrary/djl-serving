/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.wlm.util;

import java.util.concurrent.atomic.AtomicInteger;

/**
 * Class to generate an unique worker id.
 *
 * @author erik.bamberg@web.de
 */
public class AutoIncIdGenerator {

    private AtomicInteger counter;
    private String prefix;

    /**
     * Constructs an {@link AutoIncIdGenerator}.
     *
     * @param prefix the prefix for the set of IDs
     */
    public AutoIncIdGenerator(String prefix) {
        this.prefix = prefix;
        counter = new AtomicInteger(1);
    }

    /**
     * Generates a new worker id.
     *
     * @return returns a new id.
     */
    public String generate() {
        return String.format("%s%04d", prefix, counter.getAndIncrement());
    }

    /**
     * Generates a new worker id without the prefix.
     *
     * @return returns a new id without the prefix.
     */
    public int generateNum() {
        return counter.getAndIncrement();
    }

    /**
     * Removes the prefix to a generated Id.
     *
     * @param id the prefixed id
     * @return the num value of the id
     */
    public int stripPrefix(String id) {
        return Integer.parseInt(id.substring(prefix.length()));
    }
}
