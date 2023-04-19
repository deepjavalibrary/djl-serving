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
package ai.djl.serving.wlm;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.wlm.util.WlmOutOfMemoryException;
import ai.djl.util.NeuronUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.MemoryUsage;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/** A class represent a loaded model and it's metadata. */
public final class ModelInfo<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);

    private static final Pattern PATTERN = Pattern.compile("MemAvailable:\\s+(\\d+) kB");

    private transient String id;
    private String version;
    private String modelUrl;
    private String engineName;
    private String loadOnDevices;

    private int queueSize;
    private int batchSize;
    private int maxBatchDelayMillis;
    private int maxIdleSeconds;
    private int minWorkers = -1;
    private int maxWorkers = -1;

    private Map<String, Object> arguments = new ConcurrentHashMap<>();
    private Map<String, String> options = new ConcurrentHashMap<>();

    transient Path modelDir;
    private transient String artifactName;
    transient Path downloadS3Dir;

    transient Properties prop;
    private transient Status status;

    private transient Class<I> inputClass;
    private transient Class<O> outputClass;
    private transient Criteria<I, O> criteria;
    private transient Map<Device, ZooModel<I, O>> models;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param modelUrl the model Url
     */
    @SuppressWarnings("unchecked")
    public ModelInfo(String modelUrl) {
        this.id = modelUrl;
        this.modelUrl = modelUrl;
        this.inputClass = (Class<I>) Input.class;
        this.outputClass = (Class<O>) Output.class;
    }

    /**
     * Constructs a {@link ModelInfo} based on a {@link Criteria}.
     *
     * @param id the id for the created {@link ModelInfo}
     * @param modelUrl the model Url
     * @param criteria the model criteria
     */
    public ModelInfo(String id, String modelUrl, Criteria<I, O> criteria) {
        this.id = id;
        this.modelUrl = modelUrl;
        this.criteria = criteria;
        inputClass = criteria.getInputClass();
        outputClass = criteria.getOutputClass();
    }

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param id the ID of the model that will be used by workflow
     * @param modelUrl the model url
     * @param version the version of the model
     * @param engineName the engine to load the model
     * @param loadOnDevices the devices to load the model on
     * @param inputClass the model input class
     * @param outputClass the model output class
     * @param queueSize the maximum request queue size
     * @param maxIdleSeconds the initial maximum idle time for workers
     * @param maxBatchDelayMillis the initial maximum delay when scaling up before giving up
     * @param batchSize the batch size for this model
     * @param minWorkers the minimum number of workers
     * @param maxWorkers the maximum number of workers
     */
    public ModelInfo(
            String id,
            String modelUrl,
            String version,
            String engineName,
            String loadOnDevices,
            Class<I> inputClass,
            Class<O> outputClass,
            int queueSize,
            int maxIdleSeconds,
            int maxBatchDelayMillis,
            int batchSize,
            int minWorkers,
            int maxWorkers) {
        this.id = id;
        this.modelUrl = modelUrl;
        this.version = version;
        this.engineName = engineName;
        this.loadOnDevices = loadOnDevices;
        this.inputClass = inputClass;
        this.outputClass = outputClass;
        this.maxBatchDelayMillis = maxBatchDelayMillis;
        this.maxIdleSeconds = maxIdleSeconds; // default max idle time 60s
        this.queueSize = queueSize;
        this.batchSize = batchSize;
        this.minWorkers = minWorkers;
        this.maxWorkers = maxWorkers;
    }

    /**
     * Loads the model to the specified device.
     *
     * @param device the device to load model on
     * @throws IOException if failed to read model file
     * @throws ModelException if failed to load the specified model
     */
    @SuppressWarnings("unchecked")
    public void load(Device device) throws ModelException, IOException {
        if (getModels().containsKey(device)) {
            return;
        }

        try {
            // Download the model again if the model files are deleted
            initialize();
            checkAvailableMemory(device);
        } catch (IOException e) {
            throw new ModelNotFoundException(e);
        }

        try {
            Criteria.Builder<I, O> builder;
            if (criteria != null) {
                builder = criteria.toBuilder();
            } else {
                builder =
                        Criteria.builder()
                                .setTypes(inputClass, outputClass)
                                .optModelUrls(modelUrl)
                                .optEngine(engineName)
                                .optArguments(arguments)
                                .optOptions(options);
                if (batchSize > 1) {
                    builder.optArgument("batchifier", "stack");
                }
            }
            logger.info("Loading model {} on {}", id, device);
            if ("nc".equals(device.getDeviceType()) && "PyTorch".equals(engineName)) {
                // assume neuron only support PyTorch
                logger.info("Bypass NC core allocation");
            } else {
                builder.optDevice(device);
            }
            if (downloadS3Dir != null) {
                // override model_id
                builder.optOption("model_id", downloadS3Dir.toAbsolutePath().toString());
            }

            ZooModel<I, O> m = builder.build().loadModel();
            models.put(device, m);
            status = Status.READY;
        } finally {
            if (status == null) {
                status = Status.FAILED;
            }
        }
    }

    /**
     * Returns all loaded models.
     *
     * @return all loaded models
     */
    public Map<Device, ZooModel<I, O>> getModels() {
        if (models == null) {
            models = new ConcurrentHashMap<>();
        }
        return models;
    }

    /**
     * Returns the loaded {@link ZooModel} for a device.
     *
     * @param device the device to return the model on
     * @return the loaded {@link ZooModel}
     */
    public ZooModel<I, O> getModel(Device device) {
        if (getModels().get(device) == null) {
            throw new IllegalStateException("Model \"" + id + "\" has not been loaded yet.");
        }
        return getModels().get(device);
    }

    /**
     * Sets the model ID.
     *
     * @param id the model ID
     */
    public void setId(String id) {
        this.id = id;
    }

    /**
     * Returns the model ID.
     *
     * @return the model ID
     */
    public String getId() {
        return id;
    }

    /**
     * Returns the model version.
     *
     * @return the model version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the engine name.
     *
     * @return the engine name
     */
    public String getEngineName() {
        return engineName;
    }

    /**
     * Returns the model url.
     *
     * @return the model url
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /**
     * Returns the model loading status.
     *
     * @return the model loading status
     */
    public Status getStatus() {
        if (status == null) {
            return Status.PENDING;
        } else if (status == Status.FAILED) {
            return Status.FAILED;
        }
        for (Model m : getModels().values()) {
            if (Boolean.parseBoolean(m.getProperty("failed"))) {
                return Status.FAILED;
            }
        }
        return status;
    }

    /**
     * Returns the model input class.
     *
     * @return the model input class
     */
    public Class<I> getInputClass() {
        return inputClass;
    }

    /**
     * Returns the model output class.
     *
     * @return the model output class
     */
    public Class<O> getOutputClass() {
        return outputClass;
    }

    /**
     * Clarifies the input and output class when not specified.
     *
     * <p>Warning: This is intended for internal use with reflection.
     *
     * @param inputClass the model input class
     * @param outputClass the model output class
     */
    public void hasInputOutputClass(Class<I> inputClass, Class<O> outputClass) {
        if (this.inputClass != null || this.outputClass != null) {
            throw new IllegalStateException(
                    "hasInputOutputClass can only be used when input or output are not yet set");
        }
        this.inputClass = inputClass;
        this.outputClass = outputClass;
    }

    /**
     * Sets the configured max idle time in seconds of workers.
     *
     * @param maxIdleSeconds the configured max idle time in seconds of workers
     */
    public void setMaxIdleSeconds(int maxIdleSeconds) {
        this.maxIdleSeconds = maxIdleSeconds;
    }

    /**
     * Returns the configured max idle time in seconds of workers.
     *
     * @return the max idle time in seconds
     */
    public int getMaxIdleSeconds() {
        return maxIdleSeconds;
    }

    /**
     * Sets the configured batch size.
     *
     * @param batchSize the configured batch size
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the configured batch size.
     *
     * @return the configured batch size
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the maximum delay in milliseconds to aggregate a batch.
     *
     * @param maxBatchDelayMillis the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelayMillis(int maxBatchDelayMillis) {
        this.maxBatchDelayMillis = maxBatchDelayMillis;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelayMillis() {
        return maxBatchDelayMillis;
    }

    /**
     * Sets the configured size of the workers queue.
     *
     * @param queueSize the configured size of the workers queue
     */
    public void setQueueSize(int queueSize) {
        this.queueSize = queueSize;
    }

    /**
     * Returns the configured size of the workers queue.
     *
     * @return requested size of the workers queue.
     */
    public int getQueueSize() {
        return queueSize;
    }

    /**
     * Returns the minimum number of workers.
     *
     * @return the minimum number of workers
     */
    public int getMinWorkers() {
        return minWorkers;
    }

    /**
     * Returns the maximum number of workers.
     *
     * @return the maximum number of workers
     */
    public int getMaxWorkers() {
        return maxWorkers;
    }

    /**
     * Initialize the model.
     *
     * @throws IOException if failed to download model
     * @throws ModelNotFoundException if model not found
     */
    public void initialize() throws IOException, ModelException {
        downloadModel();
        loadServingProperties();
        if (engineName == null) {
            engineName = inferEngine();
        }
        downloadS3();
        // override prop keys are not write to serving.properties,
        // we have to explicitly set in Criteria
        for (String key : prop.stringPropertyNames()) {
            if (key.startsWith("option.")) {
                options.put(key.substring(7), prop.getProperty(key));
            } else {
                arguments.put(key, prop.getProperty(key));
            }
        }
    }

    /** Close all loaded models. */
    public void close() {
        if (!getModels().isEmpty() && !Boolean.getBoolean("ai.djl.serving.keep_cache")) {
            logger.info("Unloading model: {}{}", id, version == null ? "" : '/' + version);
            if (downloadS3Dir != null) {
                Utils.deleteQuietly(downloadS3Dir);
            }
            Path path = null;
            for (Model m : models.values()) {
                m.close();
                path = m.getModelPath();
            }
            models.clear();
            Path cacheDir = Utils.getCacheDir().toAbsolutePath();
            if (Objects.requireNonNull(path).startsWith(cacheDir)) {
                Utils.deleteQuietly(path);
            }
        }
    }

    /**
     * Infer model name form model URL in case model name is not provided.
     *
     * @param url the model URL
     * @return the model name
     */
    public static String inferModelNameFromUrl(String url) {
        URI uri = URI.create(url);
        String path = uri.getPath();
        if (path == null) {
            path = uri.getSchemeSpecificPart();
        }
        boolean isDirectory = path.endsWith("/");
        if (isDirectory) {
            path = path.substring(0, path.length() - 1);
        }
        int pos = path.lastIndexOf('/');
        String modelName;
        if (pos >= 0) {
            modelName = path.substring(pos + 1);
        } else {
            modelName = path;
        }
        if (!isDirectory) {
            modelName = FilenameUtils.getNamePart(modelName);
        }
        modelName = modelName.replaceAll("(\\W|^_)", "_");
        return modelName;
    }

    /**
     * Returns the default device for this model if device is null.
     *
     * @param deviceName the device to use if it is not null
     * @return a non-null device
     */
    public Device withDefaultDevice(String deviceName) {
        return Device.fromName(deviceName, Engine.getEngine(engineName));
    }

    private String inferEngine() throws ModelException {
        String engine = prop.getProperty("engine");
        if (engine != null) {
            return engine;
        }

        String prefix = prop.getProperty("option.modelName", artifactName);
        if (isPythonModel(prefix)) {
            return LmiUtils.inferLmiEngine(this);
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".pt"))
                || Files.isRegularFile(modelDir.resolve("model.pt"))) {
            return "PyTorch";
        } else if (Files.isRegularFile(modelDir.resolve("config.pbtxt"))) {
            return "TritonServer";
        } else if (Files.isRegularFile(modelDir.resolve("saved_model.pb"))) {
            return "TensorFlow";
        } else if (Files.isRegularFile(modelDir.resolve(prefix + "-symbol.json"))) {
            return "MXNet";
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".onnx"))
                || Files.isRegularFile(modelDir.resolve("model.onnx"))) {
            return "OnnxRuntime";
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".trt"))
                || Files.isRegularFile(modelDir.resolve(prefix + ".uff"))) {
            return "TensorRT";
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".tflite"))) {
            return "TFLite";
        } else if (Files.isRegularFile(modelDir.resolve("model"))
                || Files.isRegularFile(modelDir.resolve("__model__"))
                || Files.isRegularFile(modelDir.resolve("inference.pdmodel"))) {
            return "PaddlePaddle";
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".json"))) {
            return "XGBoost";
        } else {
            try {
                if (Utils.getCurrentEpoch(modelDir, prefix) >= 0) {
                    // Assume this is DJL model
                    return Engine.getDefaultEngineName();
                }
            } catch (IOException e) {
                logger.warn("Failed search parameter files in folder: " + modelDir, e);
            }
        }
        throw new ModelNotFoundException("Failed to detect engine of the model: " + modelDir);
    }

    private boolean isPythonModel(String prefix) {
        return Files.isDirectory(modelDir.resolve("MAR-INF"))
                || Files.isRegularFile(modelDir.resolve("model.py"))
                || Files.isRegularFile(modelDir.resolve(prefix + ".py"))
                || Utils.getEnvOrSystemProperty("HF_MODEL_ID") != null
                || Files.isRegularFile(modelDir.resolve("config.json"));
    }

    private void downloadModel() throws ModelNotFoundException, IOException {
        Repository repository = Repository.newInstance("modelStore", modelUrl);
        List<MRL> mrls = repository.getResources();
        if (mrls.isEmpty()) {
            throw new ModelNotFoundException("Invalid model url: " + modelUrl);
        }

        Artifact artifact = mrls.get(0).getDefaultArtifact();
        repository.prepare(artifact);
        modelDir = Utils.getNestedModelDir(repository.getResourceDirectory(artifact));
        artifactName = artifact.getName();
    }

    private void loadServingProperties() {
        if (prop == null) {
            Path file = modelDir.resolve("serving.properties");
            prop = new Properties();
            if (Files.isRegularFile(file)) {
                try (InputStream is = Files.newInputStream(file)) {
                    prop.load(is);
                } catch (IOException e) {
                    logger.warn("Failed read serving.properties file", e);
                }
            }
            configPerModelSettings();
        }
    }

    private void configPerModelSettings() {
        // per model settings can only be configured once
        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        if (queueSize <= 0) {
            queueSize = intValue(prop, "job_queue_size", wlmc.getJobQueueSize());
        }
        if (batchSize <= 0) {
            batchSize = intValue(prop, "batch_size", wlmc.getBatchSize());
        }
        if (maxBatchDelayMillis <= 0) {
            maxBatchDelayMillis = intValue(prop, "max_batch_delay", wlmc.getMaxBatchDelayMillis());
        }
        if (maxIdleSeconds <= 0) {
            maxIdleSeconds = intValue(prop, "max_idle_time", wlmc.getMaxIdleSeconds());
        }
        if (loadOnDevices == null) {
            loadOnDevices = prop.getProperty("load_on_devices", wlmc.getLoadOnDevices());
        }
        logger.info(
                "Apply per model settings:\n\tqueueSize: {}\n\tbatchSize: {}"
                        + "\n\tmaxBatchDelay: {}\n\tmaxIdle: {}\n\tloadOnDevices: {}"
                        + "\n\tengine: {}\n\tentrypoint: {}",
                queueSize,
                batchSize,
                maxBatchDelayMillis,
                maxIdleSeconds,
                loadOnDevices,
                engineName,
                prop.get("option.entryPoint"));
    }

    void checkAvailableMemory(Device device) throws IOException {
        if (Boolean.getBoolean("skip_oom_check")) {
            return;
        }

        long requiredMemory = intValue(prop, "required_memory_mb", 0) * 1024L * 1024;
        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        int defMemory = wlmc.getReservedMemoryMb();
        long reservedMemory = intValue(prop, "reserved_memory_mb", defMemory) * 1024L * 1024;
        int tpDegree = Integer.parseInt(Utils.getenv("TENSOR_PARALLEL_DEGREE", "0"));
        tpDegree = intValue(prop, "option.tensor_parallel_degree", tpDegree);
        if (requiredMemory <= 0
                && tpDegree < 1
                && "true".equals(Utils.getenv("SAGEMAKER_MULTI_MODEL"))) {
            // TODO: handle LMI use case in future
            logger.warn("No reserved_memory_mb defined, estimating memory usage ...");
            try (Stream<Path> walk = Files.walk(modelDir)) {
                requiredMemory =
                        walk.filter(Files::isRegularFile).mapToLong(ModelInfo::getFileSize).sum();
            }
            if (downloadS3Dir != null) {
                try (Stream<Path> walk = Files.walk(downloadS3Dir)) {
                    requiredMemory +=
                            walk.filter(Files::isRegularFile)
                                    .mapToLong(ModelInfo::getFileSize)
                                    .sum();
                }
            }
            // estimate the memory to be 1.2x of file size
            requiredMemory = requiredMemory * 12 / 10;
        }
        // Assume requires the same amount of CPU memory when load on GPU
        long free = getAvailableCpuMemory();
        logger.info(
                "Available CPU memory: {} MB, required: {} MB, reserved: {} MB",
                free / 1024 / 1024,
                requiredMemory / 1024 / 1024,
                reservedMemory / 1024 / 1024);
        if (free - requiredMemory < reservedMemory) {
            throw new WlmOutOfMemoryException("No enough memory to load the model.");
        }

        if (device.isGpu()) {
            MemoryUsage usage = CudaUtils.getGpuMemory(device);
            free = usage.getMax() - usage.getCommitted();
            long gpuMem = intValue(prop, "gpu.reserved_memory_mb", -1) * 1024L * 1024;
            if (gpuMem > 0) {
                reservedMemory = gpuMem;
            }
            gpuMem = intValue(prop, "gpu.required_memory_mb", -1) * 1024L * 1024;
            if (gpuMem > 0) {
                requiredMemory = gpuMem;
            }
            logger.info(
                    "Available GPU memory: {} MB, required: {} MB, reserved: {} MB",
                    free / 1024 / 1024,
                    requiredMemory / 1024 / 1024,
                    reservedMemory / 1024 / 1024);
            if (free - requiredMemory < reservedMemory) {
                throw new WlmOutOfMemoryException("No enough memory to load the model.");
            }
        }
    }

    /**
     * Returns the devices the model will be loaded on at startup.
     *
     * @return the devices the model will be loaded on at startup
     */
    public String[] getLoadOnDevices() {
        Engine engine = Engine.getEngine(engineName);
        if ("*".equals(loadOnDevices)) {
            int gpuCount = engine.getGpuCount();
            if (gpuCount > 0) {
                if ("Python".equals(engineName)) {
                    String v = Utils.getenv("TENSOR_PARALLEL_DEGREE", "-1");
                    v = prop.getProperty("option.tensor_parallel_degree", v);
                    int tensorParallelDegree = Integer.parseInt(v);
                    if (tensorParallelDegree > 0) {
                        int procs = gpuCount / tensorParallelDegree;
                        if (procs == 0) {
                            throw new EngineException(
                                    "GPU devices are not enough to run "
                                            + tensorParallelDegree
                                            + " partitions.");
                        }
                        gpuCount = procs;
                    }
                } else if ("DeepSpeed".equals(engineName)
                        || "FasterTransformer".equals(engineName)) {
                    return new String[] {"0"};
                }

                return IntStream.range(0, gpuCount)
                        .mapToObj(String::valueOf)
                        .toArray(String[]::new);
            } else if (NeuronUtils.hasNeuron()) {
                int neurons = NeuronUtils.getNeuronCores();
                String v = Utils.getenv("TENSOR_PARALLEL_DEGREE", "-1");
                v = prop.getProperty("option.tensor_parallel_degree", v);
                int tensorParallelDegree = Integer.parseInt(v);
                if (tensorParallelDegree > 0) {
                    // Assume user understand TP only works on inf2
                    int procs = neurons / tensorParallelDegree;
                    if (procs == 0) {
                        throw new EngineException(
                                "Neuron devices are not enough to run "
                                        + tensorParallelDegree
                                        + " partitions. Please refer to: "
                                        + "https://github.com/aws-neuron/transformers-neuronx#tensor-parallelism-support");
                    }
                    neurons = procs;
                }
                return IntStream.range(0, neurons).mapToObj(i -> "nc" + i).toArray(String[]::new);
            }
        } else if (!loadOnDevices.isEmpty()) {
            return loadOnDevices.split(";");
        }
        return new String[] {"-1"};
    }

    private static long getFileSize(Path path) {
        try {
            return Files.size(path);
        } catch (IOException e) {
            logger.warn("Failed to get size of: " + path, e);
        }
        return 0L;
    }

    private long getAvailableCpuMemory() {
        if (System.getProperty("os.name").startsWith("Linux")) {
            try (Scanner scanner = new Scanner(Paths.get("/proc/meminfo"))) {
                while (scanner.hasNext()) {
                    String line = scanner.nextLine();
                    Matcher m = PATTERN.matcher(line);
                    if (m.matches()) {
                        return Long.parseLong(m.group(1)) * 1024;
                    }
                }
                logger.warn("Failed to read free memory from /proc/meminfo");
            } catch (IOException e) {
                logger.warn("Failed open /proc/meminfo file", e);
            }
        }
        return Integer.MAX_VALUE * 1024L;
    }

    void downloadS3() throws ModelException, IOException {
        String s3Url = prop.getProperty("option.s3url");
        String modelId = prop.getProperty("option.model_id");
        if (s3Url != null) {
            // s3url is deprecated, use model_id instead
            if (modelId != null) {
                throw new MalformedModelException("model_id and s3url could not both set!");
            }
            modelId = s3Url;
        }
        if (modelId == null || !modelId.startsWith("s3://")) {
            return;
        }

        logger.info("S3 url found, start downloading from {}", modelId);
        // Use fixed download path to avoid repeat download
        String hash = Utils.hash(modelId);
        String downloadDir = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = downloadDir == null ? Utils.getCacheDir() : Paths.get(downloadDir);
        parent = parent.resolve("download");
        downloadS3Dir = parent.resolve(hash);
        if (Files.exists(downloadS3Dir)) {
            logger.info("artifacts has been downloaded already: {}", downloadS3Dir);
            return;
        }
        Files.createDirectories(parent);
        Path tmp = Files.createTempDirectory(parent, "tmp");
        try {
            downloadS3(modelId, tmp.toAbsolutePath().toString());
            Utils.moveQuietly(tmp, downloadS3Dir);
            logger.info("Download completed! Files saved to {}", downloadS3Dir);
        } finally {
            Utils.deleteQuietly(tmp);
        }
    }

    private void downloadS3(String src, String dest) throws ModelException {
        try {
            String[] commands;
            if (Files.exists(Paths.get("/opt/djl/bin/s5cmd"))) {
                if (!src.endsWith("*")) {
                    if (src.endsWith("/")) {
                        src = src + '*';
                    } else {
                        src = src + "/*";
                    }
                }
                commands =
                        new String[] {
                            "/opt/djl/bin/s5cmd", "--retry-count", "1", "sync", src, dest
                        };
            } else {
                logger.info("s5cmd is not installed, using aws cli");
                commands = new String[] {"aws", "s3", "sync", src, dest};
            }
            Process exec = new ProcessBuilder(commands).redirectErrorStream(true).start();
            String logOutput;
            try (InputStream is = exec.getInputStream()) {
                logOutput = Utils.toString(is);
            }
            int exitCode = exec.waitFor();
            if (0 != exitCode || logOutput.startsWith("ERROR ")) {
                logger.error(logOutput);
                throw new EngineException("Download model failed.");
            } else {
                logger.debug(logOutput);
            }
        } catch (IOException | InterruptedException e) {
            throw new ModelNotFoundException("Model failed to download from s3", e);
        }
    }

    private static int intValue(Properties prop, String key, int defValue) {
        String value = prop.getProperty(key);
        if (value == null) {
            return defValue;
        }
        return Integer.parseInt(value);
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ModelInfo)) {
            return false;
        }
        ModelInfo<?, ?> modelInfo = (ModelInfo<?, ?>) o;
        return id.equals(modelInfo.id) && Objects.equals(version, modelInfo.version);
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(id, version);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return id + ':' + version;
        }
        return id;
    }

    /** An enum represents state of a model. */
    public enum Status {
        PENDING,
        READY,
        FAILED
    }
}
