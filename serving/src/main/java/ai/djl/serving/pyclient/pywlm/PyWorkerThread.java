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
import ai.djl.serving.pyclient.protocol.RequestType;
import ai.djl.serving.util.CodecUtils;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Scanner;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.concurrent.atomic.AtomicBoolean;

public class PyWorkerThread implements Runnable {
    private final Logger logger = LoggerFactory.getLogger(PyWorkerThread.class);

    private Channel nettyClient;
    private LinkedBlockingDeque<PyJob> jobQueue;

    public Channel getNettyClient() {
        return nettyClient;
    }

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
     * @param request   to be sent
     * @param resFuture response future gets completed when response is received
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


    public PyWorkerThread(Builder builder) {
        this.jobQueue = builder.jobQueue;
        this.nettyClient = builder.nettyClient;
    }

    public static class Builder {
        private String socketPath;
        private String host;
        private int port;
        private String pythonPath;
        private Channel nettyClient;
        private LinkedBlockingDeque<PyJob> jobQueue;
        private PythonConnector connector;

        /**
         * Sets the uds socket path.
         *
         * @param socketPath uds socket path
         * @return this Builder
         */
        public Builder setSocketPath(String socketPath) {
            this.socketPath = socketPath;
            return this;
        }

        /**
         * Sets the host for TCP connection with python server.
         *
         * @param host host of the python server
         * @return this Builder
         */
        public Builder setHost(String host) {
            this.host = host;
            return self();
        }

        protected Builder self() {
            return this;
        }

        /**
         * Sets the port for TCP connection with python server.
         *
         * @param port port of the python server
         * @return this Builder
         */
        public Builder setPort(int port) {
            this.port = port;
            return self();
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

        public Builder setPythonConnector(PythonConnector connector) {
            this.connector = connector;
            return self();
        }

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

            System.out.println(Arrays.toString(args));
            try {
                Process process = Runtime.getRuntime().exec(args);
                CompletableFuture<Boolean> future = new CompletableFuture<>();
                ReaderThread readerThread = new ReaderThread("Python server", process.getInputStream(), future);
                readerThread.start();

                if (future.get()) {
                    this.nettyClient = connector.connect();
                    return new PyWorkerThread(this);
                }
            } catch (InterruptedException | IOException | ExecutionException e) {
                e.printStackTrace();
            }
            return null;
        }
    }


    private static final class ReaderThread extends Thread {

        private InputStream is;
        private AtomicBoolean isRunning = new AtomicBoolean(true);
        private CompletableFuture<Boolean> isServerStartedFuture;
        private Logger logger = LoggerFactory.getLogger(ReaderThread.class);

        public ReaderThread(String name, InputStream is, CompletableFuture<Boolean> isServerStartedFuture) {
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
                logger.info("Py" +  isRunning);

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
            } finally {
                logger.info("Stopped Scanner - {}", getName());
                try {
                    is.close();
                } catch (IOException e) {
                    logger.error("Failed to close stream for thread {}", this.getName(), e);
                }
            }
        }
    }
}