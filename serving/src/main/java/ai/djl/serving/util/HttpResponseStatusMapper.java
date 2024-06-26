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

import io.netty.handler.codec.http.HttpResponseStatus;

/** An interface that allows customizing the HttpResponseStatus to return to the client. */
public interface HttpResponseStatusMapper {

    /**
     * Returns the HttpResponseStatus to be returned to the client based on the exception
     *
     * <p>This method is called by the InferenceRequestHandler on exceptions received during
     * inference.
     *
     * @param throwable the exception received from executing inference.
     * @return the HttpResponseStatus that should be returned for the given exception.
     */
    HttpResponseStatus getHttpStatusForException(Throwable throwable);
}
