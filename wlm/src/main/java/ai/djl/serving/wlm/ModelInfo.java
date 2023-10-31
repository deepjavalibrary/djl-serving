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
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.wlm.util.WlmOutOfMemoryException;
import ai.djl.translate.TranslateException;
import ai.djl.util.NeuronUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import com.google.gson.JsonParseException;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.lang.management.MemoryUsage;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** A class represent a loaded model and it's metadata. */
public final class ModelInfo<I, O> extends WorkerPoolConfig<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private static final Pattern PATTERN = Pattern.compile("MemAvailable:\\s+(\\d+) kB");

    private String engineName;
    private String loadOnDevices;

    // the following fields can be loaded from workflow json file
    private Map<String, String> filters;
    private Map<String, Object> arguments;
    private Map<String, String> options;
    private String application;
    private String modelName;
    private String translatorFactory;
    private String translator;

    private boolean dynamicAdapters;

    transient Path modelDir;
    private transient String artifactName;
    transient Path downloadS3Dir;

    transient Properties prop;
    private transient Status status;

    private transient Class<I> inputClass;
    private transient Class<O> outputClass;
    private transient Criteria<I, O> criteria;
    private transient Map<Device, ZooModel<I, O>> models;
    private transient Map<String, Adapter> adapters;
    private transient Engine engine;
    private transient boolean initialize;

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

        adapters = new ConcurrentHashMap<>();
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

        adapters = new ConcurrentHashMap<>();
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
        this.minWorkers = Math.min(minWorkers, maxWorkers);
        this.maxWorkers = maxWorkers;

        adapters = new ConcurrentHashMap<>();
    }

    /**
     * Performs post workflow parsing initialization.
     *
     * @param workflowDir the workflow parent directory
     */
    public void postWorkflowParsing(String workflowDir) {
        if (modelUrl == null) {
            throw new JsonParseException("modelUrl is required in workflow definition.");
        }
        modelUrl = modelUrl.replaceAll("\\{model_dir}", workflowDir);
    }

    /** {@inheritDoc} */
    @Override
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
                                .optModelName(modelName)
                                .optEngine(engineName)
                                .optFilters(filters)
                                .optArguments(arguments)
                                .optOptions(options);
                if (application != null) {
                    builder.optArgument("application", application);
                }
                if (translator != null) {
                    builder.optArgument("translator", translator);
                }
                if (translatorFactory != null) {
                    builder.optArgument("translatorFactory", translatorFactory);
                }
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
            if (engine == null) {
                engine = m.getNDManager().getEngine();
            }

            if (models.isEmpty()) {
                // Check for adapters on first load
                if (Files.isDirectory(modelDir.resolve("adapters"))) {
                    Files.list(modelDir.resolve("adapters"))
                            .forEach(
                                    adapterDir -> {
                                        String adapterName = adapterDir.getFileName().toString();
                                        Adapter adapter =
                                                Adapter.newInstance(
                                                        this,
                                                        adapterName,
                                                        adapterDir.toAbsolutePath().toString());
                                        registerAdapter(adapter);
                                    });
                }
            }

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

    /** {@inheritDoc} */
    @Override
    public ThreadConfig<I, O> newThread(Device device) {
        return new ModelThread(device);
    }

    /**
     * Returns the engine.
     *
     * @return the engine
     */
    public Engine getEngine() {
        return engine;
    }

    /**
     * Returns the engine name.
     *
     * @return the engine name
     */
    public String getEngineName() {
        return engineName;
    }

    /** {@inheritDoc} */
    @Override
    public Status getStatus() {
        if (status == null) {
            return Status.PENDING;
        } else if (status == Status.FAILED) {
            return Status.FAILED;
        }

        for (Model m : getModels().values()) {
            int failures = Integer.parseInt(m.getProperty("failed", "0"));
            if (failures > 0) {
                String def = Utils.getenv("SERVING_RETRY_THRESHOLD", "10");
                int threshold = Integer.parseInt(m.getProperty("retry_threshold", def));
                if (failures > threshold) {
                    logger.info("exceed retry threshold: {}, mark model as failed.", threshold);
                    return Status.FAILED;
                }
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

    /** {@inheritDoc} */
    @Override
    public int getMinWorkers(Device device) {
        if (minWorkers != null && minWorkers >= 0) {
            return minWorkers;
        }

        return getWorkersMinMaxProperty(getModel(device), device, "minWorkers", 1);
    }

    /** {@inheritDoc} */
    @Override
    public int getMaxWorkers(Device device) {
        if (maxWorkers != null && maxWorkers >= 0) {
            return maxWorkers;
        }

        WlmConfigManager configManager = WlmConfigManager.getInstance();
        if (configManager.isDebug()) {
            return 1;
        }
        // get from model's property
        Model model = getModel(device);
        int maxProp = getWorkersMinMaxProperty(model, device, "maxWorkers", -1);
        if (maxProp > 0) {
            return maxProp;
        }

        NDManager manager = model.getNDManager();
        if ("nc".equals(device.getDeviceType())) {
            if ("Python".equals(manager.getEngine().getEngineName())) {
                return 1;
            }
            return 2; // default to max 2 workers for inferentia
        }

        if (Device.Type.GPU.equals(device.getDeviceType())) {
            String eng = manager.getEngine().getEngineName();
            if ("MXNet".equals(eng) || "Python".equals(eng)) {
                // FIXME: MXNet GPU Model doesn't support multi-threading
                return 1;
            }
            return 2;
        }

        int cpuCores = Runtime.getRuntime().availableProcessors();
        int ompThreads = Integer.parseInt(Utils.getenv("OMP_NUM_THREADS", "-1"));
        if (ompThreads > 0) {
            if (ompThreads > cpuCores) {
                ompThreads = cpuCores;
            }
            return cpuCores / ompThreads;
        }
        return 2;
    }

    private int getWorkersMinMaxProperty(Model model, Device device, String key, int def) {
        String workers = model.getProperty(device.getDeviceType() + '.' + key);
        if (workers != null) {
            return Integer.parseInt(workers);
        }
        workers = model.getProperty(key);
        if (workers != null) {
            return Integer.parseInt(workers);
        }
        return def;
    }

    /** {@inheritDoc} */
    @Override
    public void initialize() throws IOException, ModelException {
        if (initialize) {
            return;
        }
        if (adapters == null) {
            adapters = new ConcurrentHashMap<>();
        }
        downloadModel();
        loadServingProperties();
        downloadS3();
        // override prop keys are not write to serving.properties,
        // we have to explicitly set in Criteria
        if (options == null) {
            options = new ConcurrentHashMap<>();
        }
        if (arguments == null) {
            arguments = new ConcurrentHashMap<>();
        }
        for (String key : prop.stringPropertyNames()) {
            if (key.startsWith("option.")) {
                options.put(key.substring(7), prop.getProperty(key));
            } else {
                arguments.put(key, prop.getProperty(key));
            }
        }
        initialize = true;
    }

    /**
     * Adds an adapter to this {@link ModelInfo}.
     *
     * @param adapter the adapter to add
     */
    public void registerAdapter(Adapter adapter) {
        synchronized (this) {
            if (adapters.containsKey(adapter.getName())) {
                throw new IllegalArgumentException(
                        "The adapter "
                                + adapter.getName()
                                + " already exists. If you want to replace it, please unregistering"
                                + " before registering a new adapter with the same name.");
            }
            adapters.put(adapter.getName(), adapter);
        }
    }

    /**
     * Removes an adapter from this {@link ModelInfo}.
     *
     * @param name the adapter to remove
     * @return the removed adapter
     */
    public Adapter unregisterAdapter(String name) {
        synchronized (this) {
            // TODO: Remove from current workers
            if (!adapters.containsKey(name)) {
                throw new IllegalArgumentException(
                        "The adapter "
                                + name
                                + " was not found and therefore can't be unregistered");
            }
            return adapters.remove(name);
        }
    }

    /**
     * Returns the adapters for this model.
     *
     * @return the adapters for this model
     */
    public Map<String, Adapter> getAdapters() {
        return adapters;
    }

    /**
     * Returns an adapter on this {@link ModelInfo}.
     *
     * @param name the adapter name to get
     * @return the adapter
     */
    public Adapter getAdapter(String name) {
        return adapters.get(name);
    }

    /** {@inheritDoc} */
    @Override
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

    /** {@inheritDoc} */
    @Override
    public Device withDefaultDevice(String deviceName) {
        return Device.fromName(deviceName, Engine.getEngine(engineName));
    }

    private String inferEngine() throws ModelException {
        String eng = prop.getProperty("engine");
        if (eng != null) {
            return eng;
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
        } else if (Files.isRegularFile(modelDir.resolve(prefix + ".json"))
                || Files.isRegularFile(modelDir.resolve(prefix + ".xgb"))
                || Files.isRegularFile(modelDir.resolve("model.xgb"))) {
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
                || prop.getProperty("option.model_id") != null
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

    private void loadServingProperties() throws ModelException {
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

    private void configPerModelSettings() throws ModelException {
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
        if (engineName == null) {
            engineName = inferEngine();
        }

        // load default settings from env
        for (Map.Entry<String, String> entry : Utils.getenv().entrySet()) {
            String key = entry.getKey();
            if (key.startsWith("OPTION_")) {
                key = key.substring(7).toLowerCase(Locale.ROOT);
                if ("entrypoint".equals(key)) {
                    key = "entryPoint";
                }
                prop.putIfAbsent("option." + key, entry.getValue());
            }
        }

        StringBuilder sb = new StringBuilder();
        for (Map.Entry<Object, Object> entry : prop.entrySet()) {
            String key = entry.getKey().toString();
            if (!"job_queue_size".equals(key)
                    && !"batch_size".equals(key)
                    && !"max_idle_time".equals(key)
                    && !"max_batch_delay".equals(key)
                    && !"load_on_devices".equals(key)
                    && !"engine".equals(key)
                    && !"option.entryPoint".equals(key)) {
                sb.append("\n\t").append(key).append(": ").append(entry.getValue());
            }
        }

        logger.info(
                "Apply per model settings:\n\tjob_queue_size: {}\n\tbatch_size: {}"
                        + "\n\tmax_batch_delay: {}\n\tmax_idle_time: {}\n\tload_on_devices: {}"
                        + "\n\tengine: {}\n\toption.entryPoint: {}{}",
                queueSize,
                batchSize,
                maxBatchDelayMillis,
                maxIdleSeconds,
                loadOnDevices,
                engineName,
                prop.get("option.entryPoint"),
                sb);
    }

    void checkAvailableMemory(Device device) throws IOException {
        if (Boolean.getBoolean("skip_oom_check")) {
            return;
        }

        long requiredMemory = intValue(prop, "required_memory_mb", 0) * 1024L * 1024;
        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        int defMemory = wlmc.getReservedMemoryMb();
        long reservedMemory = intValue(prop, "reserved_memory_mb", defMemory) * 1024L * 1024;
        String tpDegreeStr = Utils.getenv("TENSOR_PARALLEL_DEGREE", "0");
        tpDegreeStr = prop.getProperty("option.tensor_parallel_degree", tpDegreeStr);
        int tpDegree;
        if ("max".equals(tpDegreeStr)) {
            Engine eng = Engine.getEngine(engineName);
            if (eng.getGpuCount() > 0) {
                tpDegree = eng.getGpuCount();
            } else {
                tpDegree = NeuronUtils.getNeuronCores();
            }
        } else {
            tpDegree = Integer.parseInt(tpDegreeStr);
        }
        if (requiredMemory <= 0
                && tpDegree < 1
                && "true".equals(Utils.getenv("SAGEMAKER_MULTI_MODEL"))) {
            // TODO:
            // 1. handle LMI use case in future
            // 2. if huggingface model_id is specified, the model is downloaded
            // in the python process, current file size based estimation doesn't work
            logger.warn("No reserved_memory_mb defined, estimating memory usage ...");
            try (Stream<Path> walk = Files.walk(modelDir)) {
                requiredMemory = walk.mapToLong(ModelInfo::getFileSize).sum();
            }
            if (downloadS3Dir != null) {
                try (Stream<Path> walk = Files.walk(downloadS3Dir)) {
                    requiredMemory += walk.mapToLong(ModelInfo::getFileSize).sum();
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

    /** {@inheritDoc} */
    @Override
    public String[] getLoadOnDevices() {
        Engine eng = Engine.getEngine(engineName);
        if ("*".equals(loadOnDevices)) {
            int gpuCount = eng.getGpuCount();
            String v = Utils.getenv("TENSOR_PARALLEL_DEGREE", "-1");
            v = prop.getProperty("option.tensor_parallel_degree", v);
            int tpDegree;
            if ("max".equals(v)) {
                if (gpuCount > 0) {
                    tpDegree = gpuCount;
                } else {
                    tpDegree = NeuronUtils.getNeuronCores();
                }
            } else {
                tpDegree = Integer.parseInt(v);
            }
            if (gpuCount > 0) {
                int gpuPerWorker = 1;
                if ("Python".equals(engineName)) {
                    if (tpDegree > 0) {
                        gpuPerWorker = tpDegree;
                        int procs = gpuCount / gpuPerWorker;
                        if (procs == 0) {
                            throw new EngineException(
                                    "GPU devices are not enough to run "
                                            + gpuPerWorker
                                            + " partitions.");
                        }
                        gpuCount = procs;
                    }
                } else if ("DeepSpeed".equals(engineName) || "MPI".equals(engineName)) {
                    return new String[] {"0"};
                }

                String[] ret = new String[gpuCount];
                for (int i = 0; i < gpuCount; ++i) {
                    ret[i] = String.valueOf(i * gpuPerWorker);
                }
                return ret;
            } else if (NeuronUtils.hasNeuron()) {
                int neurons = NeuronUtils.getNeuronCores();
                int ncPerWorker;
                if (tpDegree > 0) {
                    // Assume user understand TP only works on inf2
                    ncPerWorker = tpDegree;
                    int procs = neurons / ncPerWorker;
                    if (procs == 0) {
                        throw new EngineException(
                                "Neuron devices are not enough to run "
                                        + ncPerWorker
                                        + " partitions. Please refer to: "
                                        + "https://github.com/aws-neuron/transformers-neuronx#tensor-parallelism-support");
                    }
                    neurons = procs;
                } else {
                    ncPerWorker = 1;
                }
                String[] ret = new String[neurons];
                for (int i = 0; i < neurons; ++i) {
                    ret[i] = "nc" + (i * ncPerWorker);
                }
                return ret;
            }
        } else if (!loadOnDevices.isEmpty()) {
            return loadOnDevices.split(";");
        }
        return new String[] {"-1"};
    }

    /** {@inheritDoc} */
    @Override
    public boolean isParallelLoading() {
        return Boolean.parseBoolean(prop.getProperty("option.parallel_loading"));
    }

    private static long getFileSize(Path path) {
        try {
            if (Files.isRegularFile(path) && !Files.isHidden(path)) {
                return Files.size(path);
            }
            return 0;
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
        if (modelId == null) {
            return;
        }
        if (modelId.startsWith("s3://")) {
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
        } else if (modelId.startsWith("djl://")) {
            logger.info("djl model zoo url found: {}", modelId);
            modelUrl = modelId;
            // download real model from model zoo
            downloadModel();
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

    protected class ModelThread extends ThreadConfig<I, O> {

        private Predictor<I, O> predictor;
        ZooModel<I, O> model;

        protected ModelThread(Device device) {
            super(device);
            model = getModel(device);
            predictor = model.newPredictor();

            boolean logModelMetric = Boolean.parseBoolean(model.getProperty("log_model_metric"));
            if (logModelMetric) {
                int metricsAggregation =
                        Integer.parseInt(model.getProperty("metrics_aggregation", "1000"));
                Metrics metrics = new Metrics();
                metrics.setLimit(metricsAggregation);
                metrics.setOnLimit((m, s) -> MODEL_METRIC.info("{}-{}", id, m.percentile(s, 50)));
                predictor.setMetrics(metrics);
            }

            synchronized (this) {
                for (Map.Entry<String, Adapter> adapter : adapters.entrySet()) {
                    configJobs.add(adapter.getValue().registerJob(ModelInfo.this, this).getJob());
                }
            }
        }

        @Override
        @SuppressWarnings("unchecked")
        public void run(List<Job<I, O>> jobs) throws TranslateException {
            List<Job<I, O>> validJobs = new ArrayList<>(jobs.size());
            for (Job<I, O> job : jobs) {
                if (job.getInput() instanceof Input) {
                    Input i = (Input) job.getInput();
                    if (i.isCancelled()) {
                        logger.debug("Skip cancelled job");
                        continue;
                    }
                    if (i.getContent().contains("adapter")) {
                        String adapter = i.getAsString("adapter");
                        if (!dynamicAdapters && !adapters.containsKey(adapter)) {
                            String failMessage =
                                    "The adapter " + adapter + " has not been registered";
                            Job.setFailOutput((Job<Input, Output>) job, 503, failMessage);
                            continue;
                        }
                    }
                }
                validJobs.add(job);
            }
            if (!validJobs.isEmpty()) {
                Job.runAll(validJobs, js -> predictor.batchPredict(js));
            }
        }

        /**
         * Returns the predictor.
         *
         * @return the predictor
         */
        public Predictor<I, O> getPredictor() {
            return predictor;
        }

        /** {@inheritDoc} */
        @Override
        public void close() {
            predictor.close();
        }
    }
}
