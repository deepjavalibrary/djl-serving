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
package ai.djl.serving.grpc;

import ai.djl.serving.http.Session;

import io.grpc.ForwardingServerCall;
import io.grpc.Grpc;
import io.grpc.Metadata;
import io.grpc.ServerCall;
import io.grpc.ServerCallHandler;
import io.grpc.ServerInterceptor;
import io.grpc.Status;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class GrpcInterceptor implements ServerInterceptor {

    private static final Logger logger = LoggerFactory.getLogger("ACCESS_LOG");

    /** {@inheritDoc} */
    @Override
    public <I, O> ServerCall.Listener<I> interceptCall(
            ServerCall<I, O> call, Metadata headers, ServerCallHandler<I, O> next) {
        String ip = call.getAttributes().get(Grpc.TRANSPORT_ATTR_REMOTE_ADDR).toString();
        String serviceName = call.getMethodDescriptor().getFullMethodName();
        Session session = new Session(ip, serviceName);

        return next.startCall(
                new ForwardingServerCall.SimpleForwardingServerCall<>(call) {

                    /** {@inheritDoc} */
                    @Override
                    public void close(final Status status, final Metadata trailers) {
                        session.setCode(status.getCode().value());
                        logger.info(session.toString());
                        super.close(status, trailers);
                    }
                },
                headers);
    }
}
