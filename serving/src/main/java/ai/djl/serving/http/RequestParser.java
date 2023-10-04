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
package ai.djl.serving.http;

import ai.djl.modality.Input;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.util.Utils;

import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;

/**
 * a parser for inbound request.
 *
 * @author erik.bamberg@web.de
 */
public class RequestParser {

    private static final Logger logger = LoggerFactory.getLogger(RequestParser.class);

    /**
     * parsing a request.
     *
     * @param req the full request.
     * @param decoder a decoder to decode the query string.
     * @return parsed input object.
     */
    public Input parseRequest(FullHttpRequest req, QueryStringDecoder decoder) {
        Input input = new Input();
        if (decoder != null) {
            for (Map.Entry<String, List<String>> entry : decoder.parameters().entrySet()) {
                String key = entry.getKey();
                for (String value : entry.getValue()) {
                    input.add(key, value);
                }
            }
        }

        for (Map.Entry<String, String> entry : req.headers().entries()) {
            String key = entry.getKey();
            if ("X-Amzn-SageMaker-Custom-Attributes".equalsIgnoreCase(key)) {
                String[] tokens = entry.getValue().split(";");
                for (String token : tokens) {
                    String[] pair = token.split("=", 2);
                    if (pair.length == 2) {
                        input.addProperty(pair[0].trim(), pair[1].trim());
                    }
                }
            } else {
                input.addProperty(key, entry.getValue());
            }
        }
        CharSequence contentType = HttpUtil.getMimeType(req);
        if (HttpPostRequestDecoder.isMultipart(req)
                || HttpHeaderValues.APPLICATION_X_WWW_FORM_URLENCODED.contentEqualsIgnoreCase(
                        contentType)) {
            int sizeLimit = ConfigManager.getInstance().getMaxRequestSize();
            HttpDataFactory factory = new DefaultHttpDataFactory(sizeLimit);
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
            try {
                while (form.hasNext()) {
                    NettyUtils.addFormData(form.next(), input);
                }
            } catch (HttpPostRequestDecoder.EndOfDataDecoderException ignore) {
                logger.trace("End of multipart items.");
            } finally {
                form.cleanFiles();
                form.destroy();
            }
        } else {
            byte[] content = NettyUtils.getBytes(req.content());
            input.add("data", content);
        }

        if (input.getProperties().containsKey("handler")) {
            if (!Boolean.parseBoolean(
                    Utils.getEnvOrSystemProperty("ALLOW_REQUEST_HANDLER_OVERRIDE"))) {
                throw new BadRequestException("The handler can't be overridden in a request");
            }
        }

        return input;
    }
}
