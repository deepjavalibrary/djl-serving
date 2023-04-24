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
package ai.djl.serving.translator;

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.translate.Batchifier;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;

import java.nio.charset.StandardCharsets;
import java.util.Map;

public class EchoTranslator implements ServingTranslator {

    @Override
    public void setArguments(Map<String, ?> arguments) {}

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        ctx.setAttachment("input", input);
        return null;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws TranslateException {
        Input input = (Input) ctx.getAttachment("input");
        boolean streaming = Boolean.parseBoolean(input.getAsString("stream"));
        boolean exception = Boolean.parseBoolean(input.getAsString("exception"));
        String contentType = input.getProperty("content-type", null);
        String accept = input.getProperty("accept", "*/*");
        if (!accept.contains("*")) {
            contentType = accept;
        }
        Output output = new Output();
        if (contentType != null) {
            output.addProperty("content-type", contentType);
        }
        if (streaming) {
            long delay = getLongValue(input, "delay", 1000);
            int numTokens = (int) getLongValue(input, "count", 5);
            ChunkedBytesSupplier cs = new ChunkedBytesSupplier();
            output.add(cs);
            new Thread(() -> sendToken(cs, delay, numTokens)).start();
        } else if (exception) {
            throw new TranslateException("predict failed");
        } else {
            long delay = getLongValue(input, "delay", 1000);
            if (delay > 0) {
                try {
                    Thread.sleep(delay);
                } catch (InterruptedException e) {
                    throw new RuntimeException(e);
                }
            }
            output.add(input.getData());
        }
        return output;
    }

    private long getLongValue(Input input, String key, long def) {
        String value = input.getAsString(key);
        if (value == null) {
            return def;
        }
        return Long.parseLong(value);
    }

    void sendToken(ChunkedBytesSupplier cs, long delay, int count) {
        try {
            for (int i = 0; i < count; ++i) {
                cs.appendContent(("tok_" + i + '\n').getBytes(StandardCharsets.UTF_8), false);
                Thread.sleep(delay);
            }
            cs.appendContent(("tok_" + count + '\n').getBytes(StandardCharsets.UTF_8), true);
        } catch (InterruptedException ignore) {
            // ignore
        }
    }
}
