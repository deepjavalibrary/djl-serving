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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.inference.streaming.ChunkedBytesSupplier;
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
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ThreadFactory;
import java.util.stream.Stream;

class Connection {

    private static final Logger logger = LoggerFactory.getLogger(Connection.class);

    private int port;
    private SocketAddress socketAddress;
    private Channel channel;
    private RequestHandler requestHandler;

    Connection(PyEnv pyEnv, int basePort, int rank, String hostname) {
        requestHandler = new RequestHandler();
        port = 19000 + basePort;
        socketAddress = getSocketAddress(pyEnv.isMpiMode(), rank, hostname);
    }

    static Process startPython(PyEnv pyEnv, Model model, int workerId, int port, String[] hosts)
            throws IOException {
        Path tmp = Paths.get(System.getProperty("java.io.tmpdir"));
        try (Stream<Path> stream = Files.list(tmp)) {
            stream.forEach(
                    p -> {
                        try {
                            String name = p.toFile().getName();
                            if (name.startsWith("djl_sock." + port) && name.endsWith(".pid")) {
                                long pid = Long.parseLong(Files.readString(p));
                                Optional<ProcessHandle> handle = ProcessHandle.of(pid);
                                if (handle.isPresent()) {
                                    logger.warn("Kill dangling process: {}", pid);
                                    handle.get().destroyForcibly();
                                }
                                Utils.deleteQuietly(p);
                            }
                        } catch (IOException e) {
                            logger.warn("", e);
                        }
                    });
        }
        File modelPath = model.getModelPath().toFile();
        String[] args = getPythonStartCmd(pyEnv, model, workerId, port, hosts);
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

    static String[] getPythonStartCmd(
            PyEnv pyEnv, Model model, int workerId, int port, String[] hosts) {
        Device device = model.getNDManager().getDevice();
        int deviceId = device.getDeviceId();
        int clusterSize = PyEnv.getClusterSize();
        int tensorParallelDegree = pyEnv.getTensorParallelDegree();
        int pipelineParallelDegree = pyEnv.getPipelineParallelDegree();
        int worldSize = tensorParallelDegree * pipelineParallelDegree;
        String entryPoint = pyEnv.getEntryPoint();
        String recommendedEntryPoint = pyEnv.getRecommendedEntryPoint();
        String pythonLogLevel = pyEnv.getPythonLogLevel();

        if (PyEnv.isMultiNode()) {
            if (worldSize % clusterSize != 0) {
                throw new IllegalArgumentException(
                        "Error: Cannot use cluster size: "
                                + clusterSize
                                + "for world size (number of total GPUs): "
                                + worldSize);
            }

            int localSize = worldSize / clusterSize;

            String cudaDevices = getVisibleDevices(workerId, localSize);
            logger.info("Set before mpirun CUDA_VISIBLE_DEVICES={}", cudaDevices);
            logger.info(
                    "Received: pp degree: {} and tp depgree: {} and cluster size: {}",
                    pipelineParallelDegree,
                    tensorParallelDegree,
                    clusterSize);
            StringBuilder sb = new StringBuilder();
            boolean first = true;
            for (String host : hosts) {
                if (first) {
                    first = false;
                } else {
                    sb.append(',');
                }
                sb.append(host).append(':').append(localSize);
            }
            String[] args = new String[50];
            args[0] = "mpirun";
            args[1] = "-np";
            args[2] = String.valueOf(worldSize);
            args[3] = "--host";
            args[4] = sb.toString();
            args[5] = "--allow-run-as-root";
            args[6] = "--bind-to";
            args[7] = "none";
            args[8] = "--mca";
            args[9] = "orte_keep_fqdn_hostnames";
            args[10] = "t";
            args[11] = "--tag-output";
            args[12] = "-x";
            args[13] = "FI_PROVIDER=efa";
            args[14] = "-x";
            args[15] = "RDMAV_FORK_SAFE=1";
            args[16] = "-x";
            args[17] = "FI_EFA_USE_DEVICE_RDMA=1";
            args[18] = "-x";
            args[19] = "LD_LIBRARY_PATH";
            args[20] = "-x";
            args[21] = "PYTHONPATH";
            args[22] = "-x";
            args[23] = "CUDA_VISIBLE_DEVICES=" + cudaDevices;
            args[24] = "-x";
            args[25] = "MASTER_ADDR=" + pyEnv.getMasterAddr();
            args[26] = "-x";
            args[27] = "MKL_DYNAMIC=FALSE";
            args[28] = pyEnv.getPythonExecutable();
            args[29] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
            args[30] = "--model-dir";
            args[31] = model.getModelPath().toAbsolutePath().toString();
            args[32] = "--entry-point";
            args[33] = entryPoint == null ? "" : entryPoint;
            // TODO: Use mix of Unix and TCP sockets for local/remote processes
            args[34] = "--sock-type";
            args[35] = "tcp";
            args[36] = "--sock-name";
            args[37] = "0.0.0.0";
            args[38] = "--port";
            args[39] = String.valueOf(port);
            args[40] = "--tensor-parallel-degree";
            args[41] = String.valueOf(tensorParallelDegree);
            args[42] = "--pipeline-parallel-degree";
            args[43] = String.valueOf(pipelineParallelDegree);
            args[44] = "--cluster-size";
            args[45] = String.valueOf(clusterSize);
            args[46] = "--recommended-entry-point";
            args[47] = recommendedEntryPoint == null ? "" : recommendedEntryPoint;
            args[48] = "--log-level";
            args[49] = pythonLogLevel;
            return args;
        } else if (pyEnv.isMpiMode()) {
            String cudaDevices = getVisibleDevices(workerId, worldSize);
            logger.info("Set CUDA_VISIBLE_DEVICES={}", cudaDevices);
            String[] args = new String[46];
            args[0] = "mpirun";
            args[1] = "-np";
            args[2] = String.valueOf(worldSize);
            args[3] = "--allow-run-as-root";
            args[4] = "--bind-to";
            args[5] = "none";
            args[6] = "--mca";
            args[7] = "btl_vader_single_copy_mechanism";
            args[8] = "none";
            args[9] = "--tag-output";
            args[10] = "-x";
            args[11] = "FI_PROVIDER=efa";
            args[12] = "-x";
            args[13] = "RDMAV_FORK_SAFE=1";
            args[14] = "-x";
            args[15] = "FI_EFA_USE_DEVICE_RDMA=1";
            args[16] = "-x";
            args[17] = "LD_LIBRARY_PATH";
            args[18] = "-x";
            args[19] = "PYTHONPATH";
            args[20] = "-x";
            args[21] = "CUDA_VISIBLE_DEVICES=" + cudaDevices;
            args[22] = "-x";
            args[23] = "MASTER_ADDR=" + pyEnv.getMasterAddr();
            args[24] = "-x";
            args[25] = "MASTER_PORT=" + port;
            args[26] = "-x";
            args[27] = "MKL_DYNAMIC=FALSE";
            args[28] = pyEnv.getPythonExecutable();
            args[29] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
            args[30] = "--model-dir";
            args[31] = model.getModelPath().toAbsolutePath().toString();
            args[32] = "--entry-point";
            args[33] = entryPoint == null ? "" : entryPoint;
            args[34] = "--sock-type";
            args[35] = "unix";
            args[36] = "--sock-name";
            args[37] = getSocketPath(port);
            args[38] = "--tensor-parallel-degree";
            args[39] = String.valueOf(tensorParallelDegree);
            args[40] = "--pipeline-parallel-degree";
            args[41] = String.valueOf(pipelineParallelDegree);
            args[42] = "--recommended-entry-point";
            args[43] = recommendedEntryPoint == null ? "" : recommendedEntryPoint;
            args[44] = "--log-level";
            args[45] = pythonLogLevel;
            return args;
        }

        // TP/PP settings
        if (worldSize > 0 && device.isGpu()) {
            String cudaDevices = getVisibleDevices(deviceId, worldSize);
            deviceId = 0; // re-map logic device to 0
            pyEnv.addEnv("CUDA_VISIBLE_DEVICES", cudaDevices);
            logger.info("Set CUDA_VISIBLE_DEVICES={}", cudaDevices);
        }
        if ("nc".equals(device.getDeviceType())) {
            if (pipelineParallelDegree > 1) {
                throw new IllegalArgumentException(
                        "Error: Neuron does not currently support pipeline parallel degree > 1");
            }
            String visibleCores = getNeuronVisibleCores(deviceId, tensorParallelDegree);
            // TODO: re-map logic device once neuron fixed bug
            pyEnv.addEnv("NEURON_RT_VISIBLE_CORES", visibleCores);
            logger.info("Set NEURON_RT_VISIBLE_CORES={}", visibleCores);

            String neuronThreads = getNeuronThreads(tensorParallelDegree);
            pyEnv.addEnv("OMP_NUM_THREADS", neuronThreads);
            logger.info("Set OMP_NUM_THREADS={}", neuronThreads);
        }
        boolean uds = Epoll.isAvailable() || KQueue.isAvailable();
        String[] args = new String[18];
        args[0] = pyEnv.getPythonExecutable();
        args[1] = PyEnv.getEngineCacheDir() + "/djl_python_engine.py";
        args[2] = "--sock-type";
        args[3] = uds ? "unix" : "tcp";
        args[4] = uds ? "--sock-name" : "--port";
        args[5] = uds ? getSocketPath(port) : String.valueOf(port);
        args[6] = "--model-dir";
        args[7] = model.getModelPath().toAbsolutePath().toString();
        args[8] = "--entry-point";
        args[9] = entryPoint == null ? "" : entryPoint;
        args[10] = "--device-id";
        args[11] = String.valueOf(deviceId);
        args[12] = "--cluster-size";
        args[13] = String.valueOf(clusterSize);
        args[14] = "--recommended-entry-point";
        args[15] = recommendedEntryPoint == null ? "" : recommendedEntryPoint;
        args[16] = "--log-level";
        args[17] = pythonLogLevel;
        return args;
    }

    private static String getVisibleDevices(int deviceId, int localDevicesPerWorker) {
        StringBuilder sb = new StringBuilder(20);
        // CUDA_VISIBLE_DEVICES=0,2,3,7 TP2
        // -> 0,2 and 3,7
        if (Utils.getenv("CUDA_VISIBLE_DEVICES") != null) {
            String[] devices = Utils.getenv("CUDA_VISIBLE_DEVICES").split(",");
            sb.append(devices[deviceId]);
            for (int i = 1; i < localDevicesPerWorker; ++i) {
                sb.append(',').append(devices[deviceId + i]);
            }
        } else {
            sb.append(deviceId);
            for (int i = 1; i < localDevicesPerWorker; ++i) {
                sb.append(',').append(deviceId + i);
            }
        }

        return sb.toString();
    }

    private static String getNeuronVisibleCores(int deviceId, int tensorParallelDegree) {
        if (tensorParallelDegree > 0) {
            return deviceId + "-" + (deviceId + tensorParallelDegree - 1);
        }
        return String.valueOf(deviceId);
    }

    private static String getNeuronThreads(int tensorParallelDegree) {
        if (tensorParallelDegree > 0) {
            return String.valueOf(tensorParallelDegree * 2);
        }
        return String.valueOf(1);
    }

    void connect() throws InterruptedException, UnknownHostException {
        logger.debug("Connecting to socket: {}", socketAddress);
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
                                        .addLast("decoder", new OutputDecoder())
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

    private SocketAddress getSocketAddress(boolean mpiMode, int rank, String hostname) {
        if (PyEnv.isMultiNode()) {
            return new InetSocketAddress(hostname, port + rank);
        }
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
        if (PyEnv.isMultiNode()) {
            return new NioEventLoopGroup(new DaemonThreadFactory());
        }
        if (Epoll.isAvailable()) {
            return new EpollEventLoopGroup(new DaemonThreadFactory());
        } else if (KQueue.isAvailable()) {
            return new KQueueEventLoopGroup(new DaemonThreadFactory());
        }
        return new NioEventLoopGroup(new DaemonThreadFactory());
    }

    private static Class<? extends Channel> getClientChannel() {
        if (PyEnv.isMultiNode()) {
            return NioSocketChannel.class;
        }
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
                BytesSupplier supplier = content.valueAt(i);
                if (supplier != null) {
                    ByteBuffer bb = supplier.toByteBuffer();
                    out.writeInt(bb.remaining());
                    out.writeBytes(bb);
                } else {
                    out.writeInt(0);
                }
            }
        }
    }

    private static final class OutputDecoder extends ByteToMessageDecoder {

        private int maxBufferSize;
        private boolean hasMoreChunk;
        private ChunkedBytesSupplier data;

        OutputDecoder() {
            String val =
                    Utils.getEnvOrSystemProperty(
                            "MAX_NETTY_BUFFER_SIZE", String.valueOf(CodecUtils.MAX_BUFFER_SIZE));
            this.maxBufferSize = Integer.parseInt(val);
        }

        /** {@inheritDoc} */
        @Override
        protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
            // this index of the reader is marked,
            // so that future reads can be done from this index.
            in.markReaderIndex();
            boolean completed = false;
            try {
                if (hasMoreChunk) {
                    hasMoreChunk = in.readByte() == 1;
                    data.appendContent(CodecUtils.readBytes(in, maxBufferSize), !hasMoreChunk);
                } else {
                    int code = in.readShort();
                    String message = CodecUtils.readUtf8(in);
                    Output output = new Output(code, message);
                    int size = in.readShort();
                    for (int i = 0; i < size; ++i) {
                        output.addProperty(CodecUtils.readUtf8(in), CodecUtils.readUtf8(in));
                    }
                    int contentSize = in.readShort();
                    if (contentSize == -1) {
                        hasMoreChunk = true;
                        data = new ChunkedBytesSupplier();
                        output.add(data);
                    } else {
                        for (int i = 0; i < contentSize; ++i) {
                            String key = CodecUtils.readUtf8(in);
                            output.add(key, CodecUtils.readBytes(in, maxBufferSize));
                        }
                    }
                    out.add(output);
                }
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
