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
import ai.djl.serving.pyclient.protocol.Request;
import ai.djl.serving.util.CodecUtils;
import ai.djl.translate.Batchifier;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import java.io.IOException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** PythonTranslator connects to python server for data processing. */
public class PythonTranslator implements ServingTranslator {

    private static final Logger logger = LoggerFactory.getLogger(PythonTranslator.class);

    /** {@inheritDoc} */
    @Override
    public void setArguments(Map<String, ?> arguments) {}

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        return null;
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, Input input)
            throws IOException, TranslateException {
        CompletableFuture<byte[]> future = new CompletableFuture<>();
        Channel nettyClient = PythonConnector.getInstance().getChannel();
        // TODO: This will be changed in the following PRs
        Request request = new Request(input.getContent().get(null));
        send(nettyClient, CodecUtils.encodeRequest(request), future);

        // obtaining response
        try {
            byte[] response = future.get(); // awaits till the future is complete, if necessary

            // decoding response
            NDManager ndManager = ctx.getNDManager();
            return NDList.decode(ndManager, response);
        } catch (ExecutionException | InterruptedException e) {
            throw new TranslateException(e);
        }
    }

    /**
     * Sends data to netty client. TODO : Will be move this method to PythonWorker later
     *
     * @param nettyClient to connect with python server
     * @param data to be sent
     * @param resFuture response future gets completed when response is received
     */
    private void send(Channel nettyClient, byte[] data, CompletableFuture<byte[]> resFuture) {
        logger.info("Sending data to python server");
        ChannelFuture writeFuture = nettyClient.writeAndFlush(Unpooled.copiedBuffer(data));
        writeFuture.addListener(
                future -> {
                    if (future.isSuccess()) {
                        logger.info("Sent data to python server");
                        nettyClient
                                .pipeline()
                                .get(RequestHandler.class)
                                .setResponseFuture(resFuture);
                    }
                });
    }
}
