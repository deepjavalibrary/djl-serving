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
package ai.djl.serving.pyclient.pywlm;

import ai.djl.serving.pyclient.protocol.Request;
import java.util.concurrent.CompletableFuture;

/** A class represents a python job. */
public class PyJob {
    private Request request;
    private CompletableFuture<byte[]> resFuture;

    /**
     * Contructs a new {@code PyJob} instance.
     *
     * @param request to be sent to python server
     * @param resFuture that holds the response
     */
    public PyJob(Request request, CompletableFuture<byte[]> resFuture) {
        this.request = request;
        this.resFuture = resFuture;
    }

    /**
     * Returns the request.
     *
     * @return request
     */
    public Request getRequest() {
        return request;
    }

    /**
     * Returns the response future.
     *
     * @return reponse future.
     */
    public CompletableFuture<byte[]> getResFuture() {
        return resFuture;
    }
}
