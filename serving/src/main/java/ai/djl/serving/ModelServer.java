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

import ai.djl.engine.Engine;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.FilenameUtils;
import ai.djl.serving.http.ServerStartupException;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.plugins.FolderScanPluginManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.serving.util.ServerGroups;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;

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

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** The main entry point for model server. */
public class ModelServer {

    private static final Logger logger = LoggerFactory.getLogger(ModelServer.class);
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");
    private static final Pattern MODEL_STORE_PATTERN = Pattern.compile("(\\[?([^?]+?)]?=)?(.+)");

    private ServerGroups serverGroups;
    private List<ChannelFuture> futures = new ArrayList<>(2);
    private AtomicBoolean stopped = new AtomicBoolean(false);

    private ConfigManager configManager;

    private FolderScanPluginManager pluginManager;

    /**
     * Creates a new {@code ModelServer} instance.
     *
     * @param configManager the model server configuration
     */
    public ModelServer(ConfigManager configManager) {
        this.configManager = configManager;
        this.pluginManager = new FolderScanPluginManager(configManager);
        serverGroups = new ServerGroups(configManager);
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
                        logger.error("Failed to install dependency: " + dep, t);
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
            logger.error("Invalid configuration: {}", e.getMessage());
            System.exit(1); // NOPMD
        } catch (ParseException e) {
            printHelp(e.getMessage(), options);
            System.exit(1); // NOPMD
        } catch (Throwable t) {
            logger.error("Unexpected error", t);
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
            throws InterruptedException, IOException, GeneralSecurityException,
                    ServerStartupException {
        try {
            logger.info("Starting model server ...");
            List<ChannelFuture> channelFutures = start();
            channelFutures.get(0).sync();
        } finally {
            serverGroups.shutdown(true);
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
            throws InterruptedException, IOException, GeneralSecurityException,
                    ServerStartupException {
        stopped.set(false);

        String version = Engine.getDjlVersion();
        logger.info("Starting djl-serving: {} ...", version);
        logger.info(configManager.dumpConfigurations());
        Dimension dim = new Dimension("Version", version);
        SERVER_METRIC.info("{}", new Metric("djl-serving", 1, Unit.COUNT, dim));

        pluginManager.loadPlugins(true);

        try {
            initModelStore();
        } catch (BadWorkflowException e) {
            throw new ServerStartupException(
                    "Failed to initialize startup models and workflows", e);
        }

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

    private void initModelStore() throws IOException, BadWorkflowException {
        Set<String> startupModels = ModelManager.getInstance().getStartupWorkflows();

        String loadModels = configManager.getLoadModels();
        Path modelStore = configManager.getModelStore();
        if (loadModels == null || loadModels.isEmpty()) {
            if (modelStore == null) {
                return;
            }
            loadModels = "ALL";
        }

        ModelManager modelManager = ModelManager.getInstance();
        List<String> urls = new ArrayList<>();
        if ("NONE".equalsIgnoreCase(loadModels)) {
            // to disable load all models from model store
            return;
        } else if ("ALL".equalsIgnoreCase(loadModels)) {
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            if (Files.isDirectory(modelStore)) {
                // contains only directory or archive files
                boolean isMultiModelsDirectory =
                        Files.list(modelStore)
                                .filter(p -> !p.getFileName().toString().startsWith("."))
                                .allMatch(
                                        p ->
                                                Files.isDirectory(p)
                                                        || FilenameUtils.isArchiveFile(
                                                                p.toString()));

                if (isMultiModelsDirectory) {
                    // Check folders to see if they can be models as well
                    try (Stream<Path> stream = Files.list(modelStore)) {
                        urls.addAll(
                                stream.map(this::mapModelUrl)
                                        .filter(Objects::nonNull)
                                        .collect(Collectors.toList()));
                    }
                } else {
                    // Check if root model store folder contains a model
                    String url = mapModelUrl(modelStore);
                    if (url != null) {
                        urls.add(url);
                    }
                }
            } else {
                logger.warn("Model store path is not found: {}", modelStore);
            }
        } else {
            String[] modelsUrls = loadModels.split("[, ]+");
            urls.addAll(Arrays.asList(modelsUrls));
        }

        String huggingFaceModelId = Utils.getEnvOrSystemProperty("HF_MODEL_ID");
        if (huggingFaceModelId != null) {
            urls.add(createHuggingFaceModel(huggingFaceModelId));
        }

        for (String url : urls) {
            logger.info("Initializing model: {}", url);
            Matcher matcher = MODEL_STORE_PATTERN.matcher(url);
            if (!matcher.matches()) {
                throw new AssertionError("Invalid model store url: " + url);
            }
            String endpoint = matcher.group(2);
            String modelUrl = matcher.group(3);
            String version = null;
            String engineName = null;
            String deviceMapping = null;
            String modelName = null;
            if (endpoint != null) {
                String[] tokens = endpoint.split(":", -1);
                modelName = tokens[0];
                if (tokens.length > 1) {
                    version = tokens[1].isEmpty() ? null : tokens[1];
                }
                if (tokens.length > 2) {
                    engineName = tokens[2].isEmpty() ? null : tokens[2];
                }
                if (tokens.length > 3) {
                    deviceMapping = tokens[3];
                }
            }

            Workflow workflow;
            URI uri = WorkflowDefinition.toWorkflowUri(modelUrl);
            if (uri != null) {
                workflow = WorkflowDefinition.parse(modelName, uri).toWorkflow();
            } else {
                if (modelName == null) {
                    modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
                }
                ModelInfo<Input, Output> modelInfo =
                        new ModelInfo<>(
                                modelName,
                                modelUrl,
                                version,
                                engineName,
                                deviceMapping,
                                Input.class,
                                Output.class,
                                -1,
                                -1,
                                -1,
                                -1,
                                -1,
                                -1);
                workflow = new Workflow(modelInfo);
            }
            CompletableFuture<Void> f =
                    modelManager
                            .registerWorkflow(workflow)
                            .exceptionally(
                                    t -> {
                                        logger.error("Failed register workflow", t);
                                        Dimension dim = new Dimension("Model", workflow.getName());
                                        SERVER_METRIC.info(
                                                "{}",
                                                new Metric(
                                                        "ModelLoadingError", 1, Unit.COUNT, dim));
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
            startupModels.add(modelName);
        }
    }

    String mapModelUrl(Path path) {
        try {
            if (!Files.exists(path)
                    || Files.isHidden(path)
                    || (!Files.isDirectory(path)
                            && !FilenameUtils.isArchiveFile(path.toString()))) {
                return null;
            }

            if (Files.list(path).findFirst().isEmpty()) {
                return null;
            }

            path = Utils.getNestedModelDir(path);
            String url = path.toUri().toURL().toString();
            String modelName = ModelInfo.inferModelNameFromUrl(url);
            logger.info("Found model {}={}", modelName, url);
            return modelName + '=' + url;
        } catch (MalformedURLException e) {
            throw new AssertionError("Invalid path: " + path, e);
        } catch (IOException e) {
            logger.warn("Failed to access file: " + path, e);
            return null;
        }
    }

    private static void printHelp(String msg, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }

    private String createHuggingFaceModel(String modelId) throws IOException {
        String hash = Utils.hash(modelId);
        String downloadDir = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = downloadDir == null ? Utils.getCacheDir() : Paths.get(downloadDir);
        Path huggingFaceModelDir = parent.resolve(hash);
        String modelName = modelId.replaceAll("(\\W|^_)", "_");
        if (Files.exists(huggingFaceModelDir)) {
            logger.warn("HuggingFace Model {} already exists, use random model name", modelId);
            return modelName + '_' + RandomUtils.nextInt() + '=' + huggingFaceModelDir;
        }
        String huggingFaceModelRevision = Utils.getEnvOrSystemProperty("HF_REVISION");
        Properties huggingFaceProperties = new Properties();
        huggingFaceProperties.put("option.model_id", modelId);
        if (huggingFaceModelRevision != null) {
            huggingFaceProperties.put("option.revision", huggingFaceModelRevision);
        }
        String task = Utils.getEnvOrSystemProperty("HF_TASK");
        if (task != null) {
            huggingFaceProperties.put("option.task", task);
        }
        Files.createDirectories(huggingFaceModelDir);
        Path propertiesFile = huggingFaceModelDir.resolve("serving.properties");
        Files.createFile(propertiesFile);
        try (BufferedWriter writer = Files.newBufferedWriter(propertiesFile)) {
            huggingFaceProperties.store(writer, null);
        }
        logger.debug("Created serving.properties for model at path {}", propertiesFile);
        return modelName + '=' + huggingFaceModelDir;
    }
}
