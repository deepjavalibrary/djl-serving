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
package ai.djl.serving.http;

import ai.djl.ModelException;
import ai.djl.serving.util.ClusterConfig;
import ai.djl.serving.util.NettyUtils;
import ai.djl.util.Utils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

/** A class handling inbound HTTP requests for the cluster management API. */
public class ClusterRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(ClusterRequestHandler.class);

    private ClusterConfig config = ClusterConfig.getInstance();

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return req.uri().startsWith("/cluster/");
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        switch (segments[2]) {
            case "sshkey":
                Path home = Paths.get(System.getProperty("user.home")).resolve(".ssh");
                Path file = home.resolve("id_rsa.pub");
                if (Files.notExists(file)) {
                    sshkeygen(home.resolve("id_rsa").toString());
                }
                NettyUtils.sendFile(ctx, file, false);
                return;
            case "status":
                List<String> messages = decoder.parameters().get("message");
                if (messages.size() != 1) {
                    NettyUtils.sendJsonResponse(ctx, new StatusResponse("Invalid request"));
                    return;
                } else if (!"OK".equals(messages.get(0))) {
                    config.setError(messages.get(0));
                }
                config.countDown();
                NettyUtils.sendJsonResponse(ctx, new StatusResponse("OK"));
                return;
            default:
                throw new ResourceNotFoundException();
        }
    }

    private void sshkeygen(String rsaFile) {
        try {
            String[] commands = {"ssh-keygen", "-q", "-t", "rsa", "-N", "''", "-f", rsaFile};
            Process exec = new ProcessBuilder(commands).redirectErrorStream(true).start();
            String logOutput;
            try (InputStream is = exec.getInputStream()) {
                logOutput = Utils.toString(is);
            }
            int exitCode = exec.waitFor();
            if (0 != exitCode) {
                logger.error("Generate ssh key failed: {}", logOutput);
                config.setError(logOutput);
                throw new IllegalStateException("Generate ssh key failed");
            } else {
                logger.debug(logOutput);
            }
        } catch (IOException | InterruptedException e) {
            config.setError("Generate ssh key failed");
            throw new IllegalStateException("Generate ssh key failed", e);
        }
    }
}
