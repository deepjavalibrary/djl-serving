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
package ai.djl.serving.wlm.util;

/** Thrown when no enough memory to load the model. */
public class WlmOutOfMemoryException extends WlmException {

    static final long serialVersionUID = 1L;

    /**
     * Constructs a {@link WlmOutOfMemoryException} with the specified detail message.
     *
     * @param message the detail message
     */
    public WlmOutOfMemoryException(String message) {
        super(message);
    }
}
