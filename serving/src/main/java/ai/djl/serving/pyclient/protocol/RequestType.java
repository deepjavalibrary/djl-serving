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

/** Represents the type of the request to the python process. */
public enum RequestType {
    PREPROCESS(0, "preprocess", "Pre-process"),
    POSTPROCESS(1, "postprocess", "Post-process");

    private int reqTypeCode;
    private String defaultFuncName;
    private String displayStr;

    RequestType(int reqTypeCode, String defaultFuncName, String displayStr) {
        this.reqTypeCode = reqTypeCode;
        this.defaultFuncName = defaultFuncName;
        this.displayStr = displayStr;
    }

    /**
     * Returns the function name.
     *
     * @return functionName
     */
    public String functionName() {
        return defaultFuncName;
    }

    /**
     * Returns the display string of the request type.
     *
     * @return displayStr
     */
    public String displayStr() {
        return displayStr;
    }

    /**
     * Returns the request type code.
     *
     * @return reqTypeCode
     */
    public int reqTypeCode() {
        return reqTypeCode;
    }
}
