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
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.util.ClusterConfig;
import ai.djl.serving.util.ModelStore;
import ai.djl.serving.util.NettyUtils;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.Utils;

import io.netty.channel.ChannelHandler.Sharable;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

/** A class handling inbound HTTP requests for the cluster management API. */
@Sharable
public final class ClusterRequestHandler extends HttpRequestHandler {

    private static final ClusterRequestHandler INSTANCE = new ClusterRequestHandler();
    private static final Logger logger = LoggerFactory.getLogger(ClusterRequestHandler.class);

    private ClusterConfig config = ClusterConfig.getInstance();
    private Function<String[], ProcessBuilder> processBuilderFunction;
    private Path sshDirPath;

    /** Create a real process builder for the default instance. */
    private ClusterRequestHandler() {
        processBuilderFunction = (cmds -> new ProcessBuilder(cmds).redirectErrorStream(true));
    }

    /** A method to allow mocking the process builder. */
    protected void setProcessBuilderFunction(
            Function<String[], ProcessBuilder> processBuilderFunction) {
        this.processBuilderFunction = processBuilderFunction;
    }

    /**
     * Override the path used for the id_rsa keys. Defaults to `user.home` property. This path can
     * only be set once and cannot be set if the keygen has been run once.
     */
    public synchronized void setSshGenDir(Path sshDirPath) {
        if (this.sshDirPath != null && !this.sshDirPath.equals(sshDirPath)) {
            logger.error("Attempt to set ssh path after running keygen");
            throw new IllegalStateException("Attempt to set ssh path after running keygen");
        }
        this.sshDirPath = sshDirPath;
    }

    /** Get the singleton instance that can be reused across channels. */
    public static ClusterRequestHandler getInstance() {
        return INSTANCE;
    }

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
            case "sshpublickey":
                if (this.sshDirPath == null) {
                    setSshGenDir(Path.of(System.getProperty("user.home")).resolve(".ssh"));
                }
                Path publicKeyFile = sshDirPath.resolve("id_rsa.pub");
                synchronized (this) {
                    if (Files.notExists(publicKeyFile)) {
                        sshkeygen(sshDirPath.resolve("id_rsa").toString());
                    }
                }
                NettyUtils.sendFile(ctx, publicKeyFile, false);
                return;
            case "models":
                ModelStore modelStore = ModelStore.getInstance();
                List<Workflow> workflows = modelStore.getWorkflows();
                Map<String, String> map = new ConcurrentHashMap<>();
                for (Workflow workflow : workflows) {
                    for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
                        ModelInfo<Input, Output> model = (ModelInfo<Input, Output>) wpc;
                        map.put(model.getId(), model.getModelUrl());
                    }
                }
                NettyUtils.sendJsonResponse(ctx, map);
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
            String[] commands = {"ssh-keygen", "-q", "-t", "rsa", "-N", "", "-f", rsaFile};
            Process exec = processBuilderFunction.apply(commands).start();
            String logOutput;
            try (InputStream is = exec.getInputStream()) {
                logOutput = Utils.toString(is);
            }
            if (!exec.waitFor(60, TimeUnit.SECONDS)) {
                exec.destroy();
                throw new IllegalStateException("Generate ssh key timeout");
            }
            int exitCode = exec.exitValue();
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
