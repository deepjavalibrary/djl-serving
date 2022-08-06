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
import ai.djl.repository.FilenameUtils;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/** A class represent a loaded model and it's metadata. */
public final class ModelInfo<I, O> {

    private static final Logger logger = LoggerFactory.getLogger(ModelInfo.class);

    private transient String id;
    private String version;
    private String modelUrl;
    private String engineName;

    private int queueSize;
    private int batchSize;
    private int maxBatchDelay;
    private int maxIdleTime;

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
    private transient Map<Device, ZooModel<I, O>> model;

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

        WlmConfigManager config = WlmConfigManager.getInstance();
        queueSize = config.getJobQueueSize();
        maxIdleTime = config.getMaxIdleTime();
        batchSize = config.getBatchSize();
        maxBatchDelay = config.getMaxBatchDelay();
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
     * @param maxIdleTime the initial maximum idle time for workers.
     * @param maxBatchDelay the initial maximum delay when scaling up before giving up.
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
            int maxIdleTime,
            int maxBatchDelay,
            int batchSize) {
        this.id = id;
        this.modelUrl = modelUrl;
        this.version = version;
        this.engineName = engineName;
        this.inputClass = inputClass;
        this.outputClass = outputClass;
        this.maxBatchDelay = maxBatchDelay;
        this.maxIdleTime = maxIdleTime; // default max idle time 60s
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
                builder.optApplication(Application.of(application));
            }
            try {
                if (translator != null) {
                    Class<? extends ServingTranslator> clazz =
                            Class.forName(translator).asSubclass(ServingTranslator.class);
                    builder.optTranslator((Translator<I, O>) clazz.getConstructor().newInstance());
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
            String ncs = String.valueOf(device.getDeviceId());
            builder.optOption("env", "NEURON_RT_VISIBLE_CORES=" + ncs);
        } else {
            builder.optDevice(device);
        }

        try {
            getModels().put(device, builder.build().loadModel());
            status = Status.READY;
        } finally {
            if (status == null) {
                status = Status.FAILED;
            }
        }
    }

    /**
     * Sets a new batchSize and returns a new configured ModelInfo object. You have to
     * triggerUpdates in the {@code ModelManager} using this new model.
     *
     * @param batchSize the batchSize to set
     * @param maxBatchDelay maximum time to wait for a free space in worker queue after scaling up
     *     workers before giving up to offer the job to the queue.
     * @param maxIdleTime time a WorkerThread can be idle before scaling down this worker.
     */
    public void configureModelBatch(int batchSize, int maxBatchDelay, int maxIdleTime) {
        if (batchSize > 0) {
            this.batchSize = batchSize;
        }
        if (maxBatchDelay >= 0) {
            this.maxBatchDelay = maxBatchDelay;
        }
        if (maxIdleTime > 0) {
            this.maxIdleTime = maxIdleTime;
        }
    }

    /**
     * Returns all loaded models.
     *
     * @return all loaded models
     */
    public Map<Device, ZooModel<I, O>> getModels() {
        if (model == null) {
            model = new ConcurrentHashMap<>();
        }
        return model;
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
        return status == null ? Status.PENDING : status;
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
     * Sets the configured maxIdleTime of workers.
     *
     * @param maxIdleTime the configured maxIdleTime of workers
     */
    public void setMaxIdleTime(int maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
    }

    /**
     * Returns the configured maxIdleTime of workers.
     *
     * @return the maxIdleTime
     */
    public int getMaxIdleTime() {
        return maxIdleTime;
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
     * @param maxBatchDelay the maximum delay in milliseconds to aggregate a batch
     */
    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    /**
     * Returns the maximum delay in milliseconds to aggregate a batch.
     *
     * @return the maximum delay in milliseconds to aggregate a batch
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
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
            for (Model m : model.values()) {
                m.close();
            }
            model.clear();
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
        Engine engine = engineName != null ? Engine.getEngine(engineName) : Engine.getInstance();
        if (deviceName == null) {
            return engine.defaultDevice();
        }
        return Device.fromName(deviceName, engine);
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
