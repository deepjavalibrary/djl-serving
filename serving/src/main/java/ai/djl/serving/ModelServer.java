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
import ai.djl.MalformedModelException;
import ai.djl.engine.Engine;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.serving.http.ServerStartupException;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.plugins.FolderScanPluginManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.serving.util.NeuronUtils;
import ai.djl.serving.util.ServerGroups;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;
import ai.djl.util.Utils;
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
import java.io.InputStream;
import java.net.MalformedURLException;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
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
import java.util.stream.IntStream;
import java.util.stream.Stream;

/** The main entry point for model server. */
public class ModelServer implements AutoCloseable {

    private static final Logger logger = LoggerFactory.getLogger(ModelServer.class);
    private static final Logger SERVER_METRIC = LoggerFactory.getLogger("server_metric");
    private static final Pattern MODEL_STORE_PATTERN = Pattern.compile("(\\[?(.+?)]?=)?(.+)");

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

            ConfigManager.init(arguments);

            ConfigManager configManager = ConfigManager.getInstance();

            InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);
            new ModelServer(configManager).startAndWait();
        } catch (IllegalArgumentException e) {
            logger.error("Invalid configuration: " + e.getMessage());
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
            List<ChannelFuture> channelFutures = start();
            logger.info("Model server started.");
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

        String version = Engine.class.getPackage().getSpecificationVersion();
        logger.info("Starting djl-serving: {} ...", version);
        logger.info(configManager.dumpConfigurations());
        Dimension dim = new Dimension("Version", version);
        SERVER_METRIC.info("{}", new Metric("djl-serving", 1, Unit.COUNT, dim));

        try {
            initModelStore();
            initWorkflows();
        } catch (URISyntaxException
                | ModelNotFoundException
                | BadWorkflowException
                | MalformedModelException e) {
            throw new ServerStartupException(
                    "Failed to initialize startup models and workflows", e);
        }
        pluginManager.loadPlugins();

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
                        listener -> logger.info("{} model server stopped.", connector.getType()));

        logger.info("{} API bind to: {}", connector.getType(), connector);
        return f;
    }

    private void initModelStore() throws IOException {
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
        List<String> urls;
        if ("NONE".equalsIgnoreCase(loadModels)) {
            // to disable load all models from model store
            return;
        } else if ("ALL".equalsIgnoreCase(loadModels)) {
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            if (!Files.isDirectory(modelStore)) {
                logger.warn("Model store path is not found: {}", modelStore);
                return;
            }

            // Check folders to see if they can be models as well
            try (Stream<Path> stream = Files.list(modelStore)) {
                urls =
                        stream.map(this::mapModelUrl)
                                .filter(Objects::nonNull)
                                .collect(Collectors.toList());
            }
        } else {
            String[] modelsUrls = loadModels.split("[, ]+");
            urls = Arrays.asList(modelsUrls);
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
            Device[] devices = {null};
            String modelName;
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
                    if (engineName != null) {
                        DependencyManager.getInstance().installEngine(engineName);
                    }
                    Engine engine = Engine.getInstance();
                    devices = parseDevices(tokens[3], engine);
                }
            } else {
                modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
            }
            if (engineName == null) {
                engineName = inferEngineFromUrl(modelUrl);
            }

            WlmConfigManager wlmc = WlmConfigManager.getInstance();
            ModelInfo<Input, Output> modelInfo =
                    new ModelInfo<>(
                            modelName,
                            modelUrl,
                            version,
                            engineName,
                            Input.class,
                            Output.class,
                            wlmc.getJobQueueSize(),
                            wlmc.getMaxIdleTime(),
                            wlmc.getMaxBatchDelay(),
                            wlmc.getBatchSize());
            Workflow workflow = new Workflow(modelInfo);
            Device[] finalDevices = devices;
            CompletableFuture<Void> f =
                    modelManager
                            .registerWorkflow(workflow)
                            .thenAccept(
                                    v -> {
                                        for (Device device : finalDevices) {
                                            modelManager.scaleWorkers(workflow, device, 1, -1);
                                        }
                                    })
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

    private void initWorkflows()
            throws IOException, URISyntaxException, ModelNotFoundException, BadWorkflowException,
                    MalformedModelException {
        Set<String> startupWorkflows = ModelManager.getInstance().getStartupWorkflows();
        String loadWorkflows = configManager.getLoadWorkflows();
        if (loadWorkflows == null || loadWorkflows.isEmpty()) {
            return;
        }

        ModelManager modelManager = ModelManager.getInstance();
        String[] urls = loadWorkflows.split("[, ]+");

        for (String url : urls) {
            logger.info("Initializing workflow: {}", url);
            Matcher matcher = MODEL_STORE_PATTERN.matcher(url);
            if (!matcher.matches()) {
                throw new AssertionError("Invalid model store url: " + url);
            }
            String endpoint = matcher.group(2);
            String workflowUrlString = matcher.group(3);
            Device[] devices = {null};
            String workflowName;
            if (endpoint != null) {
                String[] tokens = endpoint.split(":", -1);
                workflowName = tokens[0];
                if (tokens.length > 1) {
                    devices = parseDevices(tokens[1], Engine.getInstance());
                }
            } else {
                workflowName = ModelInfo.inferModelNameFromUrl(workflowUrlString);
            }

            URL workflowUrl = new URL(workflowUrlString);
            Workflow workflow =
                    WorkflowDefinition.parse(workflowUrl.toURI(), workflowUrl.openStream())
                            .toWorkflow();

            Device[] finalDevices = devices;
            CompletableFuture<Void> f =
                    modelManager
                            .registerWorkflow(workflow)
                            .thenAccept(
                                    v -> {
                                        for (Device device : finalDevices) {
                                            modelManager.scaleWorkers(workflow, device, 1, -1);
                                        }
                                    })
                            .exceptionally(
                                    t -> {
                                        logger.error("Failed register workflow", t);
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
            startupWorkflows.add(workflowName);
        }
    }

    String mapModelUrl(Path path) {
        try {
            logger.info("Found file in model_store: {}", path);
            if (Files.isHidden(path)
                    || (!Files.isDirectory(path)
                            && !FilenameUtils.isArchiveFile(path.toString()))) {
                return null;
            }

            path = Utils.getNestedModelDir(path);
            String url = path.toUri().toURL().toString();
            String modelName = ModelInfo.inferModelNameFromUrl(url);
            String engine;
            if (Files.isDirectory(path)) {
                engine = inferEngine(path, path.toFile().getName());
            } else {
                engine = inferEngineFromUrl(url);
            }
            if (engine == null) {
                return null;
            }
            return modelName + "::" + engine + ':' + configManager.getLoadOnDevices() + '=' + url;
        } catch (MalformedURLException e) {
            throw new AssertionError("Invalid path: " + path, e);
        } catch (IOException e) {
            logger.warn("Failed to access file: " + path, e);
            return null;
        }
    }

    private String inferEngineFromUrl(String modelUrl) {
        try {
            Repository repository = Repository.newInstance("modelStore", modelUrl);
            List<MRL> mrls = repository.getResources();
            if (mrls.isEmpty()) {
                throw new IllegalArgumentException("Invalid model url: " + modelUrl);
            }

            Artifact artifact = mrls.get(0).getDefaultArtifact();
            repository.prepare(artifact);
            Path modelDir = repository.getResourceDirectory(artifact);
            return inferEngine(modelDir, artifact.getName());
        } catch (IOException e) {
            logger.warn("Failed to extract model: " + modelUrl, e);
            return null;
        }
    }

    private String inferEngine(Path modelDir, String modelName) {
        modelDir = Utils.getNestedModelDir(modelDir);

        Path file = modelDir.resolve("serving.properties");
        if (Files.isRegularFile(file)) {
            Properties prop = new Properties();
            try (InputStream is = Files.newInputStream(file)) {
                prop.load(is);
                String engine = prop.getProperty("engine");
                if (engine != null) {
                    return engine;
                }
                modelName = prop.getProperty("modelName", modelName);
            } catch (IOException e) {
                logger.warn("Failed read serving.properties file", e);
            }
        }
        if (Files.isDirectory(modelDir.resolve("MAR-INF"))
                || Files.isRegularFile(modelDir.resolve("model.py"))
                || Files.isRegularFile(modelDir.resolve(modelName + ".py"))) {
            // MMS/TorchServe
            return "Python";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".pt"))) {
            return "PyTorch";
        } else if (Files.isRegularFile(modelDir.resolve("saved_model.pb"))) {
            return "TensorFlow";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + "-symbol.json"))) {
            return "MXNet";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".onnx"))) {
            return "OnnxRuntime";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".trt"))
                || Files.isRegularFile(modelDir.resolve(modelName + ".uff"))) {
            return "TensorRT";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".tflite"))) {
            return "TFLite";
        } else if (Files.isRegularFile(modelDir.resolve("model"))
                || Files.isRegularFile(modelDir.resolve("__model__"))
                || Files.isRegularFile(modelDir.resolve("inference.pdmodel"))) {
            return "PaddlePaddle";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".json"))) {
            return "XGBoost";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".dylib"))
                || Files.isRegularFile(modelDir.resolve(modelName + ".so"))) {
            return "DLR";
        }
        logger.warn("Failed to detect engine of the model: " + modelDir);
        return null;
    }

    private Device[] parseDevices(String devices, Engine engine) {
        if ("*".equals(devices)) {
            int gpuCount = CudaUtils.getGpuCount();
            if (gpuCount > 0) {
                return IntStream.range(0, gpuCount).mapToObj(Device::gpu).toArray(Device[]::new);
            } else if (NeuronUtils.hasNeuron()) {
                int neurons = NeuronUtils.getNeuronCores();
                return IntStream.range(0, neurons)
                        .mapToObj(i -> Device.of("nc", i))
                        .toArray(Device[]::new);
            }

        } else if (!devices.isEmpty()) {
            return Arrays.stream(devices.split(";"))
                    .map(n -> Device.fromName(n, engine))
                    .toArray(Device[]::new);
        }
        return new Device[] {null};
    }

    private static void printHelp(String msg, Options options) {
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(msg, options);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        stop();
    }
}
