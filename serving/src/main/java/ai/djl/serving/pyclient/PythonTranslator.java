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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.serving.util.CodecUtils;
import ai.djl.translate.Batchifier;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

public class PythonTranslator implements ServingTranslator {
    private static final Logger logger = LoggerFactory.getLogger(PythonTranslator.class);

    @Override
    public void setArguments(Map<String, ?> arguments) {
    }

    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) throws Exception {
        CompletableFuture<byte[]> future = new CompletableFuture<>();
        Channel nettyClient = SocketConnector.getInstance().getChannel();
        send(nettyClient, list.encode(), future);

        //obtaining response
        byte[] response = future.get(); // awaits till the future is complete, if necessary
        return CodecUtils.decodeToOutput(response);
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) throws Exception {
        CompletableFuture<byte[]> future = new CompletableFuture<>();
        Channel nettyClient = SocketConnector.getInstance().getChannel();
        send(nettyClient, CodecUtils.encodeInput(input), future);

        //obtaining response
        byte[] response = future.get(); // awaits till the future is complete, if necessary

        // decoding response
        NDManager ndManager = ctx.getNDManager();
        return NDList.decode(ndManager, response);
    }


    /**
     * TODO : Will be move this method to PythonWorker later
     * Sends data to nettyclient
     *
     * @param nettyClient
     * @param data
     * @param resFuture
     * @throws IOException
     */
    private void send(Channel nettyClient, byte[] data, CompletableFuture<byte[]> resFuture) throws IOException {
        ChannelFuture writeFuture = nettyClient.writeAndFlush(Unpooled.copiedBuffer(data));
        writeFuture.addListener(future -> {
            if (future.isSuccess()) {
                nettyClient.pipeline()
                        .get(RequestHandler.class)
                        .setResponseFuture(resFuture);
            }
        });
    }
}
