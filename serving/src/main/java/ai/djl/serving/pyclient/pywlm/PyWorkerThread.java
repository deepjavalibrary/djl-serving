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
package ai.djl.serving.pyclient.pywlm;

import ai.djl.serving.pyclient.PythonConnector;
import ai.djl.serving.pyclient.RequestHandler;
import ai.djl.serving.pyclient.protocol.Request;
import ai.djl.serving.util.CodecUtils;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** This class represents a python worker thread. */
public class PyWorkerThread implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(PyWorkerThread.class);

    private Channel nettyClient;
    private LinkedBlockingDeque<PyJob> jobQueue;

    /**
     * Constructs a new {@code PyWorkerThread} instance.
     *
     * @param builder builder
     */
    public PyWorkerThread(Builder builder) {
        this.jobQueue = builder.jobQueue;
        this.nettyClient = builder.nettyClient;
    }

    /**
     * Returns the netty client.
     *
     * @return netty client.
     */
    public Channel getNettyClient() {
        return nettyClient;
    }

    /**
     * Returns a new {@code Builder} instance.
     *
     * @return builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        try {
            PyJob job = jobQueue.take();
            Request request = job.getRequest();
            send(request, job.getResFuture());

        } catch (Exception ex) {
            logger.info("Python server is not started");
        }
    }

    /**
     * Sends data to the netty client.
     *
     * @param request to be sent
     * @param resFuture response future gets completed when response is received
     * @throws IOException if error occurs while sending
     */
    private void send(Request request, CompletableFuture<byte[]> resFuture) throws IOException {
        byte[] requestData = CodecUtils.encodeRequest(request);
        ChannelFuture writeFuture = nettyClient.writeAndFlush(Unpooled.copiedBuffer(requestData));
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

    /** A Builder to construct a {@link PyWorkerThread}. */
    public static class Builder {
        private static final Logger logger = LoggerFactory.getLogger(Builder.class);

        private String pythonPath;
        private Channel nettyClient;
        private LinkedBlockingDeque<PyJob> jobQueue;
        private PythonConnector connector;

        protected Builder self() {
            return this;
        }

        /**
         * Sets the path of python.
         *
         * @param pythonPath path of the python
         * @return this Builder
         */
        public Builder setPythonPath(String pythonPath) {
            this.pythonPath = pythonPath;
            return self();
        }

        /**
         * Sets the jobQueue used to poll for new jobs.
         *
         * @param jobQueue jobQueue
         * @return this Builder
         */
        public Builder setJobQueue(LinkedBlockingDeque<PyJob> jobQueue) {
            this.jobQueue = jobQueue;
            return self();
        }

        /**
         * Sets the python connector.
         *
         * @param connector python connector
         * @return this Builder
         */
        public Builder setPythonConnector(PythonConnector connector) {
            this.connector = connector;
            return self();
        }

        /**
         * Builds the {@link PyWorkerThread} with the provided data after starting the python
         * server.
         *
         * @return an {@link PyWorkerThread}
         */
        public PyWorkerThread build() {
            String[] args = new String[10];
            args[0] = pythonPath;
            args[1] = "/Users/sindhuso/djl-serving/serving/src/main/python/python_server.py";
            args[2] = "--sock-type";
            args[3] = connector.isUds() ? "unix" : "tcp";
            args[4] = "--host";
            args[5] = connector.getHost();
            args[6] = "--port";
            args[7] = String.valueOf(connector.getPort());
            args[8] = "--sock-name";
            args[9] = connector.getSocketPath();

            try {
                Process process = Runtime.getRuntime().exec(args);
                CompletableFuture<Boolean> future = new CompletableFuture<>();
                ReaderThread readerThread =
                        new ReaderThread("Python server", process.getInputStream(), future);
                readerThread.start();

                if (future.get()) {
                    this.nettyClient = connector.connect();
                    return new PyWorkerThread(this);
                }
            } catch (InterruptedException | IOException | ExecutionException e) {
                logger.error("Error occurred when starting a python server ");
            }
            return null;
        }
    }

    private static final class ReaderThread extends Thread {

        private InputStream is;
        private AtomicBoolean isRunning = new AtomicBoolean(true);
        private CompletableFuture<Boolean> isServerStartedFuture;
        private Logger logger = LoggerFactory.getLogger(ReaderThread.class);

        public ReaderThread(
                String name, InputStream is, CompletableFuture<Boolean> isServerStartedFuture) {
            super(name + "-stdout");
            this.is = is;
            this.isServerStartedFuture = isServerStartedFuture;
        }

        public void terminate() {
            isRunning.set(false);
        }

        @Override
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                while (isRunning.get() && scanner.hasNext()) {

                    String result = scanner.nextLine();
                    logger.info("Py" + result);

                    if (result == null) {
                        break;
                    }

                    if ("Python server started.".equals(result)) {
                        this.isServerStartedFuture.complete(true);
                    }
                }
            } catch (Exception e) {
                logger.error("Couldn't create scanner - {}", getName(), e);
            }

            logger.info("Stopped Scanner - {}", getName());
            try {
                is.close();
            } catch (IOException e) {
                logger.error("Failed to close stream for thread {}", this.getName(), e);
            }
        }
    }
}
