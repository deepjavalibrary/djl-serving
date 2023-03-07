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

import ai.djl.Application;
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.repository.Artifact;
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.serving.wlm.util.WlmOutOfMemoryException;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;
import ai.djl.util.Pair;
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
import java.util.stream.Stream;

/** A class represent a loaded model and it's metadata. */
public final class ModelInfo<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);

    private static final Pattern PATTERN = Pattern.compile("MemAvailable:\\s+(\\d+) kB");

    private transient String id;
    private String version;
    private String modelUrl;
    private String engineName;

    private int queueSize;
    private int batchSize;
    private int maxBatchDelayMillis;
    private int maxIdleSeconds;

    private Map<String, String> filters;
    private Map<String, Object> arguments;
    private Map<String, String> options;
    private String application;
    private String modelName;
    private String translatorFactory;
    private String translator;
    private transient Status status;

    private transient Class<I> inputClass;
    private transient Class<O> outputClass;
    private transient Criteria<I, O> criteria;
    private transient Map<Device, ZooModel<I, O>> models;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param inputClass the model input class
     * @param outputClass the model output class
     * @param modelUrl the model Url
     */
    public ModelInfo(String modelUrl, Class<I> inputClass, Class<O> outputClass) {
        this.id = modelUrl;
        this.modelUrl = modelUrl;
        this.inputClass = inputClass;
        this.outputClass = outputClass;
    }

    /**
     * Constructs a {@link ModelInfo} based on a {@link Criteria}.
     *
     * @param id the id for the created {@link ModelInfo}
     * @param criteria the model criteria
     */
    public ModelInfo(String id, Criteria<I, O> criteria) {
        this.id = id;
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
     * @param inputClass the model input class
     * @param outputClass the model output class
     * @param queueSize the maximum request queue size
     * @param maxIdleSeconds the initial maximum idle time for workers.
     * @param maxBatchDelayMillis the initial maximum delay when scaling up before giving up.
     * @param batchSize the batch size for this model.
     */
    public ModelInfo(
            String id,
            String modelUrl,
            String version,
            String engineName,
            Class<I> inputClass,
            Class<O> outputClass,
            int queueSize,
            int maxIdleSeconds,
            int maxBatchDelayMillis,
            int batchSize) {
        this.id = id;
        this.modelUrl = modelUrl;
        this.version = version;
        this.engineName = engineName;
        this.inputClass = inputClass;
        this.outputClass = outputClass;
        this.maxBatchDelayMillis = maxBatchDelayMillis;
        this.maxIdleSeconds = maxIdleSeconds; // default max idle time 60s
        this.queueSize = queueSize;
        this.batchSize = batchSize;
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
            Criteria.Builder<I, O> builder;
            if (criteria != null) {
                builder = criteria.toBuilder();
            } else {
                // Download the model first, and get model specific configuration
                // batchSize is required before model loading in dynamic batching case
                try {
                    Pair<String, Path> pair = downloadModel(modelUrl);
                    checkAvailableMemory(device, pair.getValue());
                    configPerModelSettings(pair.getValue());
                } catch (IOException e) {
                    throw new ModelNotFoundException(e);
                }

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
                    builder.optApplication(Application.of(application));
                }
                try {
                    if (translator != null) {
                        Class<? extends ServingTranslator> clazz =
                                Class.forName(translator).asSubclass(ServingTranslator.class);
                        builder.optTranslator(
                                (Translator<I, O>) clazz.getConstructor().newInstance());
                    }
                    if (translatorFactory != null) {
                        Class<? extends TranslatorFactory> clazz =
                                Class.forName(translator).asSubclass(TranslatorFactory.class);
                        builder.optTranslatorFactory(clazz.getConstructor().newInstance());
                    }
                } catch (ReflectiveOperationException e) {
                    throw new ModelException("Invalid criteria", e);
                }
                if (batchSize > 1) {
                    builder.optArgument("batchifier", "stack");
                }
            }
            logger.info("Loading model {} on {}", id, device);
            if ("nc".equals(device.getDeviceType())) {
                //                String ncs = String.valueOf(device.getDeviceId());
                //                builder.optOption("env", "NEURON_RT_VISIBLE_CORES=" + ncs);
                logger.info("Bypass NC core allocation");
            } else {
                builder.optDevice(device);
            }

            ZooModel<I, O> m = builder.build().loadModel();
            if (criteria != null) {
                // TODO: user has to manually configure batchifier if using dynamic batch
                configPerModelSettings(m.getModelPath());
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

    /**
     * Sets the model ID.
     *
     * @param id the model ID
     */
    public void setModelId(String id) {
        this.id = id;
    }

    /**
     * Returns the model ID.
     *
     * @return the model ID
     */
    public String getModelId() {
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

    /** Close all loaded models. */
    public void close() {
        if (!getModels().isEmpty()) {
            logger.info("Unloading model: {}{}", id, version == null ? "" : '/' + version);
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
     * Infers engine name from model URL.
     *
     * @param modelUrl the model URL
     * @return the engine name
     */
    public static String inferEngineFromUrl(String modelUrl) {
        try {
            Pair<String, Path> pair = downloadModel(modelUrl);
            return inferEngine(pair.getValue(), pair.getKey());
        } catch (IOException e) {
            logger.warn("Failed to extract model: " + modelUrl, e);
            return null;
        }
    }

    /**
     * Infers which device to load.
     *
     * @param modelUrl the model URL
     * @return the device name
     */
    public static String inferDeviceName(String modelUrl) {
        try {
            Pair<String, Path> pair = downloadModel(modelUrl);
            Properties prop = getServingProperties(pair.getValue());
            return prop.getProperty("load_on_devices");
        } catch (IOException e) {
            logger.warn("Failed to extract model: " + modelUrl, e);
            return null;
        }
    }

    /**
     * Infers engine name from model directory.
     *
     * @param modelDir the model directory
     * @param modelName the model name
     * @return the engine name
     */
    public static String inferEngine(Path modelDir, String modelName) {
        modelDir = Utils.getNestedModelDir(modelDir);

        Properties prop = getServingProperties(modelDir);
        String engine = prop.getProperty("engine");
        if (engine != null) {
            return engine;
        }

        modelName = prop.getProperty("option.modelName", modelName);
        if (Files.isDirectory(modelDir.resolve("MAR-INF"))
                || Files.isRegularFile(modelDir.resolve("model.py"))
                || Files.isRegularFile(modelDir.resolve(modelName + ".py"))) {
            // MMS/TorchServe
            return "Python";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".pt"))
                || Files.isRegularFile(modelDir.resolve("model.pt"))) {
            return "PyTorch";
        } else if (Files.isRegularFile(modelDir.resolve("saved_model.pb"))) {
            return "TensorFlow";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + "-symbol.json"))) {
            return "MXNet";
        } else if (Files.isRegularFile(modelDir.resolve(modelName + ".onnx"))
                || Files.isRegularFile(modelDir.resolve("model.onnx"))) {
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
        }
        logger.warn("Failed to detect engine of the model: {}", modelDir);
        return null;
    }

    /**
     * Returns the default device for this model if device is null.
     *
     * @param deviceName the device to use if it is not null
     * @return a non-null device
     */
    public Device withDefaultDevice(String deviceName) {
        if (engineName == null && modelUrl != null) {
            engineName = inferEngineFromUrl(modelUrl);
        }
        if (deviceName == null && modelUrl != null) {
            deviceName = inferDeviceName(modelUrl);
        }
        Engine engine = engineName != null ? Engine.getEngine(engineName) : Engine.getInstance();
        // TODO: Load model API doesn't support * or multiple devices
        if (deviceName == null || "*".equals(deviceName)) {
            return engine.defaultDevice();
        }
        String[] devices = deviceName.split(";");
        return Device.fromName(devices[0], engine);
    }

    /**
     * Downloads model from the model URL.
     *
     * @param modelUrl the model URL
     * @return model name and downloaded model path
     * @throws IOException if failed to download the model
     */
    public static Pair<String, Path> downloadModel(String modelUrl) throws IOException {
        Repository repository = Repository.newInstance("modelStore", modelUrl);
        List<MRL> mrls = repository.getResources();
        if (mrls.isEmpty()) {
            throw new IOException("Invalid model url: " + modelUrl);
        }

        Artifact artifact = mrls.get(0).getDefaultArtifact();
        repository.prepare(artifact);
        Path modelDir = Utils.getNestedModelDir(repository.getResourceDirectory(artifact));
        return new Pair<>(artifact.getName(), modelDir);
    }

    /**
     * Loads the serving properties from model folder.
     *
     * @param modelDir model directory
     * @return the serving properties
     */
    public static Properties getServingProperties(Path modelDir) {
        Path file = modelDir.resolve("serving.properties");
        Properties prop = new Properties();
        if (Files.isRegularFile(file)) {
            try (InputStream is = Files.newInputStream(file)) {
                prop.load(is);
            } catch (IOException e) {
                logger.warn("Failed read serving.properties file", e);
            }
        }
        return prop;
    }

    private void configPerModelSettings(Path modelDir) throws IOException {
        // per model settings can only be configured once
        Properties prop = getServingProperties(modelDir);
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
    }

    void checkAvailableMemory(Device device, Path modelDir) throws IOException {
        if (Boolean.getBoolean("skip_oom_check")) {
            return;
        }

        Properties prop = getServingProperties(modelDir);
        long requiredMemory = intValue(prop, "required_memory_mb", 0) * 1024L * 1024;
        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        int defMemory = wlmc.getReservedMemoryMb();
        long reservedMemory = intValue(prop, "reserved_memory_mb", defMemory) * 1024L * 1024;
        int tpDegree = Integer.parseInt(Utils.getenv("TENSOR_PARALLEL_DEGREE", "0"));
        tpDegree = intValue(prop, "option.tensor_parallel_degree", tpDegree);
        if (requiredMemory <= 0 && tpDegree < 1) {
            // TODO: handle LMI use case in future
            try (Stream<Path> walk = Files.walk(modelDir)) {
                // estimate the memory to be 1.2x of file size
                requiredMemory =
                        walk.filter(Files::isRegularFile).mapToLong(ModelInfo::getFileSize).sum();
                requiredMemory = requiredMemory * 12 / 10;
            }
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

    private static long getFileSize(Path path) {
        try {
            return Files.size(path);
        } catch (IOException e) {
            logger.warn("Failed to get size of: " + path, e);
        }
        return 0L;
    }

    private static long getAvailableCpuMemory() {
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
