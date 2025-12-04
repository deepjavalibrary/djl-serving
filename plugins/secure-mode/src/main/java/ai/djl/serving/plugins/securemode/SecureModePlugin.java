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

import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.IllegalConfigurationException;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.wlm.util.EventManager;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/** A plugin for Secure Mode. */
public class SecureModePlugin implements RequestHandler<Void> {

    private static final Pattern ADAPTERS_PATTERN =
            Pattern.compile("^(/models/[^/^?]+)?/adapters([/?].*)?");

    /** Constructs a new {@code SecureModePlugin} instance. */
    public SecureModePlugin() {
        if (SecureModeUtils.isSecureMode()) {
            EventManager.getInstance().addListener(new SecureModeModelServerListener());
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (!SecureModeUtils.isSecureMode()) {
            return false;
        }

        if (msg instanceof FullHttpRequest) {
            FullHttpRequest req = (FullHttpRequest) msg;
            String uri = req.uri();
            // Intercept adapter registration requests
            return ADAPTERS_PATTERN.matcher(uri).matches() && HttpMethod.POST.equals(req.method());
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        // Validate adapter before allowing the request to proceed
        Map<String, List<String>> params = decoder.parameters();
        if (params.containsKey("src")) {
            String src = params.get("src").get(0);
            try {
                SecureModeAdapterValidator.validateAdapterPath(src);
            } catch (IllegalConfigurationException e) {
                throw new BadRequestException(e.getMessage(), e);
            } catch (IOException e) {
                throw new BadRequestException("Error validating adapter: " + e.getMessage(), e);
            }
        }

        // Return null to allow the request to continue to the actual handler
        return null;
    }
}
