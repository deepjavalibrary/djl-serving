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

import ai.djl.serving.GrpcServer;
import ai.djl.serving.util.ConfigManager;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.ServerInterceptors;
import io.grpc.netty.shaded.io.grpc.netty.NettyServerBuilder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.concurrent.TimeUnit;

/** {@link GrpcServer} implementation. */
public class GrpcServerImpl extends GrpcServer {

    private static final Logger logger = LoggerFactory.getLogger(GrpcServerImpl.class);

    private Server server;

    /** {@inheritDoc} */
    @Override
    public void start() throws IOException {
        ConfigManager configManager = ConfigManager.getInstance();
        String ip = configManager.getProperty("grpc_address", "127.0.0.1");
        int port = configManager.getIntProperty("grpc_port", 8082);
        InetSocketAddress address = new InetSocketAddress(ip, port);
        long maxConnectionAge =
                configManager.getIntProperty("grpc_max_connection_age", Integer.MAX_VALUE);
        long maxConnectionGrace =
                configManager.getIntProperty("grpc_max_connection_grace", Integer.MAX_VALUE);

        ServerBuilder<?> s =
                NettyServerBuilder.forAddress(address)
                        .maxConnectionAge(maxConnectionAge, TimeUnit.MILLISECONDS)
                        .maxConnectionAgeGrace(maxConnectionGrace, TimeUnit.MILLISECONDS)
                        .maxInboundMessageSize(configManager.getMaxRequestSize())
                        .addService(
                                ServerInterceptors.intercept(
                                        new InferenceService(), new GrpcInterceptor()));

        server = s.build();
        server.start();
        logger.info("gRPC bind to port: {}:{}", ip, port);
    }

    /** {@inheritDoc} */
    @Override
    public void stop() {
        if (server != null) {
            try {
                server.shutdown().awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                logger.warn("Stop gPRC server failed", e);
            }
        }
    }
}
