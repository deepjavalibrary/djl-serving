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
package ai.djl.serving.pyclient;

import ai.djl.serving.pyclient.protocol.Response;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.concurrent.CompletableFuture;

/**
 * A class handling inbound request handler for ipc with python.
 */
public class RequestHandler extends SimpleChannelInboundHandler<Response>  {
    private static final Logger logger = LoggerFactory.getLogger(RequestHandler.class);
    private CompletableFuture<byte[]> future;

    /** {@inheritDoc} */
    @Override
    protected void channelRead0(ChannelHandlerContext ctx, Response msg) {
        byte[] rawData = msg.getRawData();
        future.complete(rawData);
    }

    /** {@inheritDoc} */
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        logger.error("Exception occurred during request handler of python worker", cause);
        ctx.close();
    }

    /**
     * Sets the response future object. It gets completed when response is sent by the python server.
     *
     * @param future response future
     */
    public void setResponseFuture(CompletableFuture<byte[]> future) {
        this.future = future;
    }

}
