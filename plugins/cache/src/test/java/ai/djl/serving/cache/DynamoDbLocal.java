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
package ai.djl.serving.cache;

import com.amazonaws.services.dynamodbv2.local.main.ServerRunner;
import com.amazonaws.services.dynamodbv2.local.server.DynamoDBProxyServer;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.ServerSocket;

/** A wrapper class to avoid class loading failure when DynamoDBLocal is not installed. */
public class DynamoDbLocal {

    private static final Logger logger = LoggerFactory.getLogger(DynamoDbLocal.class);

    private DynamoDBProxyServer server;
    private int port;

    public DynamoDbLocal() {
        port = findFreePort();
    }

    public void startLocalServer() {
        String[] localArgs = {"-inMemory", "-port", String.valueOf(port)};
        try {
            server = ServerRunner.createServerFromCommandLineArgs(localArgs);
            server.start();
        } catch (Exception e) {
            logger.error("Failed to start DynamoDB", e);
        }
    }

    public void stop() {
        if (server != null) {
            try {
                server.stop();
            } catch (Exception e) {
                logger.error("Failed to stop DynamoDB", e);
            }
        }
    }

    public int getPort() {
        return port;
    }

    private static int findFreePort() {
        try (ServerSocket socket = new ServerSocket(0)) {
            return socket.getLocalPort();
        } catch (IOException ignore) {
            // ignore
        }
        return -1;
    }
}
