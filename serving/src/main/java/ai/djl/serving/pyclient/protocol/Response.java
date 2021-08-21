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

/** Response format for ipc with python server. */
public class Response {

    private byte[] rawData;

    /**
     * Getter for rawData.
     *
     * @return response data in bytes
     */
    public byte[] getRawData() {
        return rawData;
    }

    /**
     * Setter for rawData.
     *
     * @param rawData response data in bytes
     */
    public void setRawData(byte[] rawData) {
        this.rawData = rawData;
    }
}
