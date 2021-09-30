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
import java.io.File;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class Connection {

    private static final Logger logger = LoggerFactory.getLogger(Connection.class);

    private static AtomicInteger counter = new AtomicInteger(0);

    private int port;
    private boolean uds;
    private Channel channel;
    private RequestHandler requestHandler;

    Connection() {
        this.port = 19000 + counter.getAndIncrement();
        uds = Epoll.isAvailable() || KQueue.isAvailable();
        requestHandler = new RequestHandler();
    }

    Process startPython(PyEnv pyEnv, Model model) throws IOException {
        File modelPath = model.getModelPath().toFile();

        String[] args = getPythonStartCmd(pyEnv, model);
        String[] envp = pyEnv.getEnvironmentVars(model);

        return Runtime.getRuntime().exec(args, envp, modelPath);
    }

    int getPort() {
        return port;
    }

    Output send(Input input) throws ExecutionException, InterruptedException, TimeoutException {
        CompletableFuture<Output> f = new CompletableFuture<>();
        requestHandler.setResponseFuture(f);
        if (!channel.writeAndFlush(input).sync().isSuccess()) {
            throw new IllegalStateException("Failed to send data to python.");
        }
        // TODO: make this configurable
        return f.get(5, TimeUnit.MINUTES);
    }

    private String[] getPythonStartCmd(PyEnv pyEnv, Model model) {
        String[] args = new String[10];
        args[0] = pyEnv.getPythonExecutable();
        args[1] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
        args[2] = "--sock-type";
        args[3] = uds ? "unix" : "tcp";
        args[4] = uds ? "--sock-name" : "--port";
        args[5] = uds ? getSocketPath() : String.valueOf(port);
        args[6] = "--model-dir";
        args[7] = model.getModelPath().toAbsolutePath().toString();
        args[8] = "--entry-point";
        args[9] = pyEnv.getEntryPoint();
        return args;
    }

    void connect() {
        EventLoopGroup group = PyEnv.getEventLoopGroup();

        try {
            Bootstrap clientBootstrap = new Bootstrap();
            clientBootstrap
                    .group(group)
                    .channel(getClientChannel())
                    .remoteAddress(getSocketAddress())
                    .handler(
                            new ChannelInitializer<Channel>() {

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
                throw new EngineException("Connection to Python process is failed.");
            }
            channel = future.awaitUninterruptibly().channel();
        } catch (InterruptedException e) {
            logger.error("Exception occurred while creating netty client", e);
            throw new EngineException("Connection to Python process is interrupted.", e);
        }
    }

    private String getSocketPath() {
        return System.getProperty("java.io.tmpdir") + "/djl_sock." + port;
    }

    private SocketAddress getSocketAddress() {
        if (uds) {
            return new DomainSocketAddress(getSocketPath());
        }
        return new InetSocketAddress("127.0.0.1", port);
    }

    static EventLoopGroup newEventLoopGroup(int threads) {
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(threads);
        } else if (KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(threads);
        }

        return new NioEventLoopGroup(threads);
    }

    private static Class<? extends Channel> getClientChannel() {
        if (Epoll.isAvailable()) {
            return EpollDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueDomainSocketChannel.class;
        }
        return NioSocketChannel.class;
    }

    /** Cleans up the leftover resources. */
    void clean() {
        if (uds) {
            Utils.deleteQuietly(Paths.get(getSocketPath()));
        }
    }

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
            logger.error("Exception occurred during reading Output from python process", cause);
            ctx.close();
        }

        /** {@inheritDoc} */
        @Override
        public void channelInactive(ChannelHandlerContext ctx) {
            ctx.fireChannelInactive();
            if (future != null) {
                future.completeExceptionally(new IOException("Connection dropped."));
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
            } catch (IndexOutOfBoundsException | NegativeArraySizeException e) {
                logger.debug("", e);
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
}
