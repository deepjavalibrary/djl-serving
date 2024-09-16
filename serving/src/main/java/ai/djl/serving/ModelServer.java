/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.http.ServerStartupException;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.plugins.FolderScanPluginManager;
import ai.djl.serving.util.ClusterConfig;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.serving.util.ModelStore;
import ai.djl.serving.util.ServerGroups;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.cuda.CudaUtils;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.handler.ssl.SslContext;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.lang.management.MemoryUsage;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;

/** The main entry point for model server. */
public class ModelServer {

    private static final Logger logger = LoggerFactory.getLogger(ModelServer.class);
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");

    private ServerGroups serverGroups;
    private List<ChannelFuture> futures = new ArrayList<>(2);
    private AtomicBoolean stopped = new AtomicBoolean(false);

    private ConfigManager configManager;

    private FolderScanPluginManager pluginManager;

    private GrpcServer grpc;

    /**
     * Creates a new {@code ModelServer} instance.
     *
     * @param configManager the model server configuration
     */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        this.pluginManager = new FolderScanPluginManager(configManager);
        serverGroups = new ServerGroups(configManager);
        grpc = GrpcServer.newInstance();
    }

    /**
     * The entry point for the model server.
     *
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        Options options = Arguments.getOptions();
        try {
            DefaultParser parser = new DefaultParser();
            CommandLine cmd = parser.parse(options, args, null, false);
            Arguments arguments = new Arguments(cmd);
            if (arguments.hasHelp()) {
                printHelp("djl-serving [OPTIONS]", options);
                return;
            } else if (cmd.hasOption("install")) {
                String[] dependencies = cmd.getOptionValues("install");
                DependencyManager dm = DependencyManager.getInstance();
                for (String dep : dependencies) {
                    try {
                        dm.installDependency(dep);
                    } catch (Throwable t) {
                        logger.error("Failed to install dependency: {}", dep, t);
                        System.exit(1); // NOPMD
                    }
                }
                return;
            }

            DependencyManager.getInstance().initialize();

            ConfigManager.init(arguments);

            ConfigManager configManager = ConfigManager.getInstance();

            InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
            new ModelServer(configManager).startAndWait();
        } catch (IllegalArgumentException e) {
            logger.error("Invalid configuration", e);
            SERVER_METRIC.info("{}", new Metric("ConfigurationError", 1));
            System.exit(1); // NOPMD
        } catch (ParseException e) {
            printHelp(e.getMessage(), options);
            SERVER_METRIC.info("{}", new Metric("CmdError", 1));
            System.exit(1); // NOPMD
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
            SERVER_METRIC.info("{}", new Metric("StartupFailed", 1));
            System.exit(1); // NOPMD
        }
    }

    /**
     * Starts the model server and block until server stops.
     *
     * @throws InterruptedException if interrupted
     * @throws IOException if failed to start socket listener
     * @throws GeneralSecurityException if failed to read SSL certificate
     * @throws ServerStartupException if failed to startup server
     */
    public void startAndWait()
            throws InterruptedException,
                    IOException,
                    GeneralSecurityException,
                    ServerStartupException {
        try {
            logger.info("Starting model server ...");
            List<ChannelFuture> channelFutures = start();
            channelFutures.get(0).sync();
        } finally {
            serverGroups.shutdown(true);
            if (grpc != null) {
                grpc.stop();
            }
            logger.info("Model server stopped.");
        }
    }

    /**
     * Main Method that prepares the future for the channel and sets up the ServerBootstrap.
     *
     * @return a list of ChannelFuture object
     * @throws InterruptedException if interrupted
     * @throws IOException if failed to start socket listener
     * @throws GeneralSecurityException if failed to read SSL certificate
     * @throws ServerStartupException if failed to startup server
     */
    public List<ChannelFuture> start()
            throws InterruptedException,
                    IOException,
                    GeneralSecurityException,
                    ServerStartupException {
        long begin = System.nanoTime();

        String version = Engine.getDjlVersion();
        logger.info("Starting djl-serving: {} ...", version);
        logger.info(configManager.dumpConfigurations());
        Dimension dim = new Dimension("Version", version);
        SERVER_METRIC.info("{}", new Metric("DJLServingStart", 1, Unit.COUNT, dim));

        pluginManager.loadPlugins(true);

        try {
            ModelStore modelStore = ModelStore.getInstance();
            modelStore.initialize();

            List<Workflow> workflows = modelStore.getWorkflows();

            initMultiNode(workflows);

            loadModels(workflows);
        } catch (BadWorkflowException | ModelException | CompletionException e) {
            throw new ServerStartupException(
                    "Failed to initialize startup models and workflows", e);
        }

        stopped.set(false);
        Connector inferenceConnector =
                configManager.getConnector(Connector.ConnectorType.INFERENCE);
        Connector managementConnector =
                configManager.getConnector(Connector.ConnectorType.MANAGEMENT);
        inferenceConnector.clean();
        managementConnector.clean();

        EventLoopGroup serverGroup = serverGroups.getServerGroup();
        EventLoopGroup workerGroup = serverGroups.getChildGroup();

        futures.clear();
        if (inferenceConnector.equals(managementConnector)) {
            Connector both = configManager.getConnector(Connector.ConnectorType.BOTH);
            futures.add(initializeServer(both, serverGroup, workerGroup));
        } else {
            futures.add(initializeServer(inferenceConnector, serverGroup, workerGroup));
            futures.add(initializeServer(managementConnector, serverGroup, workerGroup));
        }

        if (grpc != null) {
            grpc.start();
        }

        long duration = (System.nanoTime() - begin) / 1000;
        Metric metric = new Metric("StartupLatency", duration, Unit.MICROSECONDS);
        SERVER_METRIC.info("{}", metric);
        for (int i = 0; i < CudaUtils.getGpuCount(); ++i) {
            try {
                Device device = Device.gpu(i);
                MemoryUsage mem = CudaUtils.getGpuMemory(device);
                SERVER_METRIC.info(
                        "{}", new Metric("GPUMemory_" + i, mem.getCommitted(), Unit.BYTES));
            } catch (IllegalArgumentException | EngineException e) {
                logger.warn("Failed get GPU memory", e);
                break;
            }
        }

        if (stopped.get()) {
            // check if model load failed in wait loading model case
            stop();
        }

        return futures;
    }

    /**
     * Return if the server is running.
     *
     * @return {@code true} if the server is running
     */
    public boolean isRunning() {
        return !stopped.get();
    }

    /** Stops the model server. */
    public void stop() {
        logger.info("Stopping model server.");
        stopped.set(true);
        for (ChannelFuture future : futures) {
            future.channel().close();
        }
        serverGroups.shutdown(true);
        serverGroups.reset();
        if (grpc != null) {
            grpc.stop();
        }
    }

    private void initMultiNode(List<Workflow> workflows)
            throws GeneralSecurityException,
                    IOException,
                    InterruptedException,
                    ServerStartupException,
                    ModelException {
        ClusterConfig cc = ClusterConfig.getInstance();
        int clusterSize = cc.getClusterSize();
        if (clusterSize > 1) {
            Connector multiNodeConnector =
                    configManager.getConnector(Connector.ConnectorType.CLUSTER);
            multiNodeConnector.clean();

            EventLoopGroup serverGroup = serverGroups.getServerGroup();
            EventLoopGroup workerGroup = serverGroups.getChildGroup();

            ChannelFuture future = initializeServer(multiNodeConnector, serverGroup, workerGroup);

            // download the models
            for (Workflow workflow : workflows) {
                for (WorkerPoolConfig<Input, Output> model : workflow.getWpcs()) {
                    model.initialize();
                }
            }
            cc.countDown();

            logger.info("Waiting for all worker nodes ready ...");
            cc.await();

            future.channel().close();
            serverGroups.shutdown(true);
            serverGroups.reset();

            String status = cc.getError();
            if (status != null) {
                throw new ServerStartupException("Failed to initialize cluster: " + status);
            }
            logger.info("Cluster initialized with {} nodes.", clusterSize);
        }
    }

    private ChannelFuture initializeServer(
            Connector connector, EventLoopGroup serverGroup, EventLoopGroup workerGroup)
            throws InterruptedException, IOException, GeneralSecurityException {
        Class<? extends ServerChannel> channelClass = connector.getServerChannel();
        logger.info(
                "Initialize {} server with: {}.",
                connector.getType(),
                channelClass.getSimpleName());

        ServerBootstrap b = new ServerBootstrap();
        b.option(ChannelOption.SO_BACKLOG, 1024)
                .channel(channelClass)
                .childOption(ChannelOption.SO_LINGER, 0)
                .childOption(ChannelOption.SO_REUSEADDR, true)
                .childOption(ChannelOption.SO_KEEPALIVE, true);
        b.group(serverGroup, workerGroup);

        SslContext sslCtx = null;
        if (connector.isSsl()) {
            sslCtx = configManager.getSslContext();
        }
        b.childHandler(new ServerInitializer(sslCtx, connector.getType(), pluginManager));

        ChannelFuture future;
        try {
            future = b.bind(connector.getSocketAddress()).sync();
        } catch (Exception e) {
            // https://github.com/netty/netty/issues/2597
            if (e instanceof IOException) {
                throw new IOException("Failed to bind to address: " + connector, e);
            }
            throw e;
        }
        future.addListener(
                (ChannelFutureListener)
                        f -> {
                            if (!f.isSuccess()) {
                                try {
                                    f.get();
                                } catch (InterruptedException | ExecutionException e) {
                                    logger.error("", e);
                                }
                                System.exit(2); // NOPMD
                            }
                            serverGroups.registerChannel(f.channel());
                        });

        future.sync();

        ChannelFuture f = future.channel().closeFuture();
        f.addListener(
                (ChannelFutureListener)
                        listener -> logger.info("{} listener stopped.", connector.getType()));

        logger.info("{} API bind to: {}", connector.getType(), connector);
        return f;
    }

    private void loadModels(List<Workflow> workflows) {
        ModelManager modelManager = ModelManager.getInstance();
        for (Workflow workflow : workflows) {
            CompletableFuture<Void> f = modelManager.registerWorkflow(workflow);
            f.exceptionally(
                    t -> {
                        logger.error("Failed register workflow", t);
                        Dimension dim = new Dimension("Model", workflow.getName());
                        SERVER_METRIC.info(
                                "{}", new Metric("ModelLoadingError", 1, Unit.COUNT, dim));
                        // delay 3 seconds, allows REST API to send PING
                        // response (health check)
                        try {
                            Thread.sleep(3000);
                        } catch (InterruptedException ignore) {
                            // ignore
                        }
                        stop();
                        return null;
                    });
            if (configManager.waitModelLoading()) {
                f.join();
            }
        }
    }

    private static void printHelp(String msg, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }
}
