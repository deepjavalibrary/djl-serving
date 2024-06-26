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

import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.wlm.util.WlmException;
import ai.djl.translate.TranslateException;

import io.netty.handler.codec.http.HttpResponseStatus;

/** Default converter translating exception to HttpResponseStatus. */
public class DefaultHttpResponseStatusMapper implements HttpResponseStatusMapper {

    /** {@inheritDoc} */
    @Override
    public HttpResponseStatus getHttpStatusForException(Throwable t) {
        HttpResponseStatus status;
        if (t instanceof TranslateException || t instanceof BadRequestException) {
            status = HttpResponseStatus.BAD_REQUEST;
        } else if (t instanceof WlmException) {
            status = HttpResponseStatus.SERVICE_UNAVAILABLE;
        } else {
            status = HttpResponseStatus.INTERNAL_SERVER_ERROR;
        }
        return status;
    }
}
