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
package ai.djl.serving.plugins.securemode;

import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.wlm.util.EventManager;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

/** A plugin for Secure Mode. */
public class SecureModePlugin implements RequestHandler<Void> {

    /** Constructs a new {@code SecureModePlugin} instance. */
    public SecureModePlugin() {
        EventManager.getInstance().addListener(new SecureModeModelServerListener());
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        return null;
    }
}
