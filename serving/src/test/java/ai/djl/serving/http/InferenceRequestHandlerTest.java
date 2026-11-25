/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Output;
import ai.djl.serving.ConfigManagerTest;
import ai.djl.serving.util.ConfigManager;

import io.netty.channel.ChannelInboundHandlerAdapter;
import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.FullHttpResponse;

import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

public class InferenceRequestHandlerTest {

    private Path configFile;

    @BeforeMethod
    public void setUp() throws IOException, ParseException {
        // one server error per hour: the first onServerError() call spends it
        configFile = Files.createTempFile("error-rate", ".properties");
        Files.writeString(configFile, "error_rate_server=1/PT1H\n");
        ConfigManager.init(
                ConfigManagerTest.parseArguments(new String[] {"-f", configFile.toString()}));
    }

    @AfterMethod
    public void tearDown() throws IOException {
        Files.deleteIfExists(configFile);
    }

    @Test
    public void testLateChunkedErrorCountsAsServerError() {
        FullHttpResponse resp = sendChunked(500);
        Assert.assertEquals(resp.status().code(), 500);
        resp.release();
        // sendOutput spent the only server-error token
        Assert.assertTrue(ConfigManager.getInstance().onServerError());
    }

    @Test
    public void testChunkedSuccessIsNotAServerError() {
        FullHttpResponse resp = sendChunked(200);
        Assert.assertEquals(resp.status().code(), 200);
        resp.release();
        Assert.assertFalse(ConfigManager.getInstance().onServerError());
    }

    // The output starts as 200 and gets its final code only when the chunk arrives, as async
    // and rolling-batch responses do.
    private static FullHttpResponse sendChunked(int finalCode) {
        ChunkedBytesSupplier data = new ChunkedBytesSupplier();
        Output output = new Output();
        output.add(data);
        CompletableFuture<Void> producer =
                CompletableFuture.runAsync(
                        () -> {
                            try {
                                TimeUnit.MILLISECONDS.sleep(200);
                            } catch (InterruptedException e) {
                                Thread.currentThread().interrupt();
                            }
                            output.setCode(finalCode);
                            data.appendContent("{}".getBytes(StandardCharsets.UTF_8), true);
                        });
        EmbeddedChannel channel = new EmbeddedChannel(new ChannelInboundHandlerAdapter());
        new InferenceRequestHandler().sendOutput(output, channel.pipeline().firstContext());
        producer.join();
        FullHttpResponse resp = channel.readOutbound();
        channel.finishAndReleaseAll();
        return resp;
    }
}
