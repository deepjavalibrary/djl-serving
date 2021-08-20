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
package ai.djl.serving.pyclient.protocol;

/**
 * Request format to python server.
 * TODO: Will be changed to support python file, method, process function type.
 */
public class Request {
    private byte[] rawData;

    /**
     * Sets the request data
     *
     * @param rawData request data in bytes
     */
    public Request(byte[] rawData) {
        this.rawData = rawData;
    }

    /**
     * Getter for rawData
     *
     * @return rawData
     */
    public byte[] getRawData() {
        return rawData;
    }

    /**
     * Setter for rawData
     *
     * @param rawData request data in bytes
     */
    public void setRawData(byte[] rawData) {
        this.rawData = rawData;
    }
}
