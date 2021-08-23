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

/** Request format for ipc with python server. */
public class Request {

    private int requestType;
    private String pythonFile;
    private String functionName;
    private byte[] functionParam;

    /**
     * Getter for requestType.
     *
     * @return type of the request
     */
    public int getRequestType() {
        return requestType;
    }

    /**
     * Sets the requestType.
     *
     * @param requestType type of the request
     * @return Request
     */
    public Request setRequestType(int requestType) {
        this.requestType = requestType;
        return this;
    }

    /**
     * Gets the pythonFile.
     *
     * @return python file
     */
    public String getPythonFile() {
        return pythonFile;
    }

    /**
     * Sets the pythonFile.
     *
     * @param pythonFile python file
     * @return Request
     */
    public Request setPythonFile(String pythonFile) {
        this.pythonFile = pythonFile;
        return this;
    }

    /**
     * Gets the functionName.
     *
     * @return functionName
     */
    public String getFunctionName() {
        return functionName;
    }

    /**
     * Sets the functionName.
     *
     * @param functionName function to be executed.
     * @return Request
     */
    public Request setFunctionName(String functionName) {
        this.functionName = functionName;
        return this;
    }

    /**
     * Gets for functionParam.
     *
     * @return functionParam
     */
    public byte[] getFunctionParam() {
        return functionParam;
    }

    /**
     * Sets the functionParam.
     *
     * @param functionParam in bytes
     * @return Request
     */
    public Request setFunctionParam(byte[] functionParam) {
        this.functionParam = functionParam;
        return this;
    }
}
