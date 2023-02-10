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
package ai.djl.python.engine;

import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.PairList;
import ai.djl.util.Utils;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.ByteToMessageDecoder;
import io.netty.handler.codec.MessageToByteEncoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.atomic.AtomicInteger;

class Connection {

    private static final Logger logger = LoggerFactory.getLogger(Connection.class);
    private static final String MASTER_ADDR = "127.0.0.1";

    private static AtomicInteger counter = new AtomicInteger(0);

    private int port;
    private SocketAddress socketAddress;
    private Channel channel;
    private RequestHandler requestHandler;

    Connection(PyEnv pyEnv, int workerId, int rank) {
        requestHandler = new RequestHandler();
        if (pyEnv.isMpiMode()) {
            port = 128 * workerId + 29761;
        } else {
            port = 19000 + counter.getAndIncrement();
        }
        socketAddress = getSocketAddress(pyEnv.isMpiMode(), rank);
    }

    static Process startPython(PyEnv pyEnv, Model model, int workerId, int port)
            throws IOException {
        File modelPath = model.getModelPath().toFile();
        String[] args = getPythonStartCmd(pyEnv, model, workerId, port);
        String[] envp = pyEnv.getEnvironmentVars(model);
        logger.debug("cmd: {}", (Object) args);

        return Runtime.getRuntime().exec(args, envp, modelPath);
    }

    int getPort() {
        return port;
    }

    CompletableFuture<Output> send(Input input) throws InterruptedException {
        CompletableFuture<Output> f = new CompletableFuture<>();
        requestHandler.setResponseFuture(f);
        if (!channel.isActive() || !channel.writeAndFlush(input).sync().isSuccess()) {
            throw new IllegalStateException("Failed to send data to python.");
        }
        return f;
    }

    static String[] getPythonStartCmd(PyEnv pyEnv, Model model, int workerId, int port) {
        int tensorParallelDegree = pyEnv.getTensorParallelDegree();
        if (pyEnv.isMpiMode()) {
            String[] args = new String[36];
            args[0] = "mpirun";
            args[1] = "-N";
            // TODO: When we support multi nodes, change it to the product of tensor parallel value
            // and
            // pipeline parallel value.
            args[2] = String.valueOf(tensorParallelDegree);
            args[3] = "--allow-run-as-root";
            args[4] = "--mca";
            args[5] = "btl_vader_single_copy_mechanism";
            args[6] = "none";
            args[7] = "--tag-output";
            args[8] = "-x";
            args[9] = "FI_PROVIDER=efa";
            args[10] = "-x";
            args[11] = "RDMAV_FORK_SAFE=1";
            args[12] = "-x";
            args[13] = "FI_EFA_USE_DEVICE_RDMA=1";
            args[14] = "-x";
            args[15] = "LD_LIBRARY_PATH";
            args[16] = "-x";
            args[17] = "PYTHONPATH";
            args[18] = "-x";
            args[19] = "CUDA_VISIBLE_DEVICES=" + getVisibleDevices(workerId, tensorParallelDegree);
            args[20] = "-x";
            args[21] = "MASTER_ADDR=" + MASTER_ADDR;
            args[22] = "-x";
            args[23] = "MASTER_PORT=" + port;
            args[24] = pyEnv.getPythonExecutable();
            args[25] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
            args[26] = "--model-dir";
            args[27] = model.getModelPath().toAbsolutePath().toString();
            args[28] = "--entry-point";
            args[29] = pyEnv.getEntryPoint();
            args[30] = "--sock-type";
            args[31] = "unix";
            args[32] = "--sock-name";
            args[33] = getSocketPath(port);
            args[34] = "--tensor-parallel-degree";
            args[35] = String.valueOf(tensorParallelDegree);
            return args;
        }

        // TP settings
        int deviceId = model.getNDManager().getDevice().getDeviceId();
        if (tensorParallelDegree > 0 && deviceId != -1) {
            String cudaDevices = getVisibleDevices(deviceId, tensorParallelDegree);
            pyEnv.addEnv("CUDA_VISIBLE_DEVICES", cudaDevices);
        }
        boolean uds = Epoll.isAvailable() || KQueue.isAvailable();
        String[] args = new String[12];
        args[0] = pyEnv.getPythonExecutable();
        args[1] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
        args[2] = "--sock-type";
        args[3] = uds ? "unix" : "tcp";
        args[4] = uds ? "--sock-name" : "--port";
        args[5] = uds ? getSocketPath(port) : String.valueOf(port);
        args[6] = "--model-dir";
        args[7] = model.getModelPath().toAbsolutePath().toString();
        args[8] = "--entry-point";
        args[9] = pyEnv.getEntryPoint();
        args[10] = "--device-id";
        args[11] = String.valueOf(deviceId);
        return args;
    }

    private static String getVisibleDevices(int workerId, int tensorParallelDegree) {
        StringBuilder sb = new StringBuilder(20);
        sb.append(workerId * tensorParallelDegree);
        for (int i = 1; i < tensorParallelDegree; ++i) {
            sb.append(',').append(workerId * tensorParallelDegree + i);
        }
        return sb.toString();
    }

    void connect() throws InterruptedException {
        EventLoopGroup group = PyEnv.getEventLoopGroup();

        Bootstrap clientBootstrap = new Bootstrap();
        clientBootstrap
                .group(group)
                .channel(getClientChannel())
                .remoteAddress(socketAddress)
                .handler(
                        new ChannelInitializer<>() {

                            @Override
                            protected void initChannel(Channel ch) {
                                ch.pipeline()
                                        .addLast("encoder", new RequestEncoder())
                                        .addLast(
                                                "decoder",
                                                new OutputDecoder(CodecUtils.MAX_BUFFER_SIZE))
                                        .addLast("handler", requestHandler);
                            }
                        });

        ChannelFuture future = clientBootstrap.connect().sync();
        if (!future.isSuccess()) {
            throw new EngineException("Connection to worker process is failed.");
        }
        channel = future.awaitUninterruptibly().channel();
    }

    void disconnect() {
        try {
            if (channel != null) {
                channel.close().sync();
            } else {
                logger.warn("Connection channel is null.");
            }
        } catch (InterruptedException ignore) {
            // ignore
        }
        if (socketAddress instanceof DomainSocketAddress) {
            String path = ((DomainSocketAddress) socketAddress).path();
            Utils.deleteQuietly(Paths.get(path));
        }
    }

    private static String getSocketPath(int port) {
        return System.getProperty("java.io.tmpdir") + "/djl_sock." + port;
    }

    private SocketAddress getSocketAddress(boolean mpiMode, int rank) {
        if (mpiMode) {
            return new DomainSocketAddress(getSocketPath(port) + '.' + rank);
        }
        boolean uds = Epoll.isAvailable() || KQueue.isAvailable();
        if (uds) {
            return new DomainSocketAddress(getSocketPath(port));
        }
        return new InetSocketAddress("127.0.0.1", port);
    }

    static EventLoopGroup newEventLoopGroup() {
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(new DaemonThreadFactory());
        } else if (KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(new DaemonThreadFactory());
        }

        return new NioEventLoopGroup(new DaemonThreadFactory());
    }

    private static Class<? extends Channel> getClientChannel() {
        if (Epoll.isAvailable()) {
            return EpollDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueDomainSocketChannel.class;
        }
        return NioSocketChannel.class;
    }

    @ChannelHandler.Sharable
    private static final class RequestHandler extends SimpleChannelInboundHandler<Output> {

        private CompletableFuture<Output> future;

        /** {@inheritDoc} */
        @Override
        protected void channelRead0(ChannelHandlerContext ctx, Output msg) {
            future.complete(msg);
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Exception reading Output from python process", cause);
            ctx.close();
        }

        /** {@inheritDoc} */
        @Override
        public void channelInactive(ChannelHandlerContext ctx) {
            ctx.fireChannelInactive();
            if (future != null) {
                future.completeExceptionally(new IOException("Python worker disconnected."));
            }
        }

        /**
         * Sets the response future object. It gets completed when response is sent by the python
         * server.
         *
         * @param future response future
         */
        public void setResponseFuture(CompletableFuture<Output> future) {
            this.future = future;
        }
    }

    private static final class RequestEncoder extends MessageToByteEncoder<Input> {

        /** {@inheritDoc} */
        @Override
        protected void encode(ChannelHandlerContext ctx, Input msg, ByteBuf out) {
            Map<String, String> prop = msg.getProperties();
            out.writeShort(prop.size());
            for (Map.Entry<String, String> entry : prop.entrySet()) {
                CodecUtils.writeUtf8(out, entry.getKey());
                CodecUtils.writeUtf8(out, entry.getValue());
            }
            PairList<String, BytesSupplier> content = msg.getContent();
            int size = content.size();
            out.writeShort(size);
            for (int i = 0; i < size; ++i) {
                CodecUtils.writeUtf8(out, content.keyAt(i));
                ByteBuffer bb = content.valueAt(i).toByteBuffer();
                out.writeInt(bb.remaining());
                out.writeBytes(bb);
            }
        }
    }

    private static final class OutputDecoder extends ByteToMessageDecoder {

        private int maxBufferSize;

        OutputDecoder(int maxBufferSize) {
            this.maxBufferSize = maxBufferSize;
        }

        /** {@inheritDoc} */
        @Override
        protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
            // this index of the reader is marked,
            // so that future reads can be done from this index.
            in.markReaderIndex();
            boolean completed = false;
            try {
                int code = in.readShort();
                String message = CodecUtils.readUtf8(in);
                Output output = new Output(code, message);
                int size = in.readShort();
                for (int i = 0; i < size; ++i) {
                    output.addProperty(CodecUtils.readUtf8(in), CodecUtils.readUtf8(in));
                }
                int contentSize = in.readShort();
                for (int i = 0; i < contentSize; ++i) {
                    String key = CodecUtils.readUtf8(in);
                    output.add(key, CodecUtils.readBytes(in, maxBufferSize));
                }
                out.add(output);
                completed = true;
            } catch (IndexOutOfBoundsException | NegativeArraySizeException ignore) {
                // ignore
            } finally {
                if (!completed) {
                    // resetting the marked index. Index will be set to 0
                    in.resetReaderIndex();
                }
            }
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            logger.error("Exception occurred during request handler of python worker", cause);
            ctx.close();
        }
    }

    private static final class DaemonThreadFactory implements ThreadFactory {

        /** {@inheritDoc} */
        @Override
        public Thread newThread(Runnable r) {
            Thread t = new Thread(r);
            t.setDaemon(true);
            return t;
        }
    }
}
