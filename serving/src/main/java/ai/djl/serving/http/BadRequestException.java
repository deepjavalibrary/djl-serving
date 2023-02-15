/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/** Thrown when a bad HTTP request is received. */
public class BadRequestException extends IllegalArgumentException {

    static final long serialVersionUID = 1L;

    private final int code;

    /**
     * Constructs an {@code BadRequestException} with the specified detail message.
     *
     * @param code the HTTP response code
     * @param message the detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     */
    public BadRequestException(int code, String message) {
        super(message);
        this.code = code;
    }

    /**
     * Constructs an {@code BadRequestException} with the specified detail message.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     */
    public BadRequestException(String message) {
        this(400, message);
    }

    /**
     * Constructs an {@code BadRequestException} with the specified detail message and a root cause.
     *
     * @param message The detail message (which is saved for later retrieval by the {@link
     *     #getMessage()} method)
     * @param cause root cause
     */
    public BadRequestException(String message, Throwable cause) {
        super(message, cause);
        this.code = 400;
    }

    /**
     * Return the HTTP response code.
     *
     * @return the HTTP response code
     */
    public int getCode() {
        return code;
    }
}
