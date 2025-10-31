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

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.BlockingDeque;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;

/** {@code PyModel} is the Python engine implementation of {@link Model}. */
public class PyModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(PyModel.class);

    private PyEnv pyEnv;
    private boolean parallelLoading;
    private BlockingDeque<PyProcess> workerQueue;

    /**
     * Constructs a new Model on a given device.
     *
     * @param name the model name
     * @param manager the {@link NDManager} to holds the NDArray
     */
    PyModel(String name, NDManager manager) {
        super(name);
        this.manager = manager;
        this.manager.setName("pythonModel");
        boolean mpiMode = ((PyEngine) manager.getEngine()).isMpiMode();
        pyEnv = new PyEnv(mpiMode);
        dataType = DataType.FLOAT32;
        workerQueue = new LinkedBlockingDeque<>();
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        setModelDir(modelPath);
        if (block != null) {
            throw new UnsupportedOperationException(
                    "Python engine does not support dynamic blocks");
        }
        String entryPoint = null;
        String recommendedEntryPoint = null;
        if (options != null) {
            // If tp_degree set to "max", we defer and set it at the end to ensure we take pp degree
            // into account.
            boolean setTensorParallelDegreeToMax = false;
            logger.debug("options in serving.properties for model: {}", modelName);
            for (Map.Entry<String, ?> entry : options.entrySet()) {
                String key = entry.getKey();
                String value = (String) entry.getValue();
                if (!"env".equals(key)) {
                    pyEnv.addParameter(key, value);
                    properties.put(key, value);
                }
                logger.debug("{}={}", key, value);
                switch (key) {
                    case "pythonExecutable":
                        pyEnv.setPythonExecutable(value);
                        break;
                    case "env":
                        String[] envs = value.split(",");
                        for (String e : envs) {
                            String[] kv = e.split("=", 2);
                            if (kv.length > 1) {
                                pyEnv.addEnv(kv[0].trim(), kv[1].trim());
                            }
                        }
                        break;
                    case "predict_timeout":
                        try {
                            int timeoutSeconds = Integer.parseInt(value);
                            pyEnv.setPredictTimeout(timeoutSeconds);
                        } catch (NumberFormatException ignore) {
                            logger.warn("Invalid predict_timeout value: {}", value);
                        }
                        break;
                    case "model_loading_timeout":
                        try {
                            int timeoutSeconds = Integer.parseInt(value);
                            pyEnv.setModelLoadingTimeout(timeoutSeconds);
                        } catch (NumberFormatException ignore) {
                            logger.warn("Invalid model_loading_timeout value: {}", value);
                        }
                        break;
                    case "entryPoint":
                        entryPoint = value;
                        break;
                    case "parallel_loading":
                        parallelLoading = Boolean.parseBoolean(value);
                        break;
                    case "tensor_parallel_degree":
                        if ("max".equals(value)) {
                            setTensorParallelDegreeToMax = true;
                        } else {
                            pyEnv.setTensorParallelDegree(Integer.parseInt(value));
                        }
                        break;
                    case "pipeline_parallel_degree":
                        if (value != null) {
                            pyEnv.setPipelineParallelDegree(Integer.parseInt(value));
                        } else {
                            pyEnv.setPipelineParallelDegree(1);
                        }
                        break;
                    case "handler":
                        pyEnv.setHandler(value);
                        break;
                    case "enable_venv":
                        pyEnv.setEnableVenv(Boolean.parseBoolean(value));
                        break;
                    case "mpi_mode":
                        pyEnv.setMpiMode(Boolean.parseBoolean(value));
                        break;
                    case "async_mode":
                        pyEnv.setAsyncMode(Boolean.parseBoolean(value));
                        break;
                    default:
                        break;
                }
            }

            if (setTensorParallelDegreeToMax) {
                int tpDegree =
                        PyEnv.getDefaultTensorParallelDegree() / pyEnv.getPipelineParallelDegree();
                pyEnv.setTensorParallelDegree(tpDegree);
            }
        }

        // MMS and TorchServe Bcc
        if (Files.isDirectory(modelDir.resolve("MAR-INF"))) {
            pyEnv.setFailOnInitialize(false);
        }

        if (entryPoint == null) {
            entryPoint = Utils.getenv("DJL_ENTRY_POINT");
            if (entryPoint == null) {
                Path modelFile = findModelFile(prefix);
                String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
                // find default entryPoint
                if (modelFile != null) {
                    entryPoint = modelFile.toFile().getName();
                }
                // find recommendedEntryPoint
                if (hasModelFile(
                        modelDir, prefix, ".skops", ".joblib", ".pkl", ".pickle", ".cloudpkl")) {
                    recommendedEntryPoint = "djl_python.sklearn_handler";
                } else if ("trtllm".equals(features)) {
                    recommendedEntryPoint = "djl_python.tensorrt_llm";
                } else if ("vllm".equals(features)) {
                    if (pyEnv.isAsyncMode()) {
                        recommendedEntryPoint = "djl_python.lmi_vllm.vllm_async_service";
                    } else {
                        recommendedEntryPoint = "djl_python.huggingface";
                    }
                } else if (pyEnv.getInitParameters().containsKey("model_id")
                        || Files.exists(modelPath.resolve("config.json"))) {
                    recommendedEntryPoint = "djl_python.huggingface";
                }
                if (entryPoint == null && recommendedEntryPoint == null) {
                    throw new FileNotFoundException(".py file not found in: " + modelPath);
                }
            }
        } else if (entryPoint.toLowerCase(Locale.ROOT).startsWith("http")) {
            String hash = Utils.hash(entryPoint);
            Path dir = Utils.getCacheDir().resolve("tmp").resolve(hash);
            Path modelFile = dir.resolve("model.py");
            if (Files.exists(modelFile)) {
                logger.info("entryPoint file already exist: {}", dir);
            } else {
                logger.info("downloading entryPoint file: {}", entryPoint);
                Files.createDirectories(dir);
                Path tmp = Files.createTempFile(dir, "download", ".tmp");
                try (InputStream is = Utils.openUrl(entryPoint)) {
                    Files.copy(is, tmp, StandardCopyOption.REPLACE_EXISTING);
                    Utils.moveQuietly(tmp, modelFile);
                } finally {
                    Utils.deleteQuietly(tmp);
                }
            }
            entryPoint = modelFile.toAbsolutePath().toString();
        }
        pyEnv.setEntryPoint(entryPoint);
        pyEnv.setRecommendedEntryPoint(recommendedEntryPoint);
        if (pyEnv.isEnableVenv()) {
            pyEnv.createVirtualEnv(Utils.hash(modelDir.toString()));
        }

        if (pyEnv.isMpiMode()) {
            int tpDegree = pyEnv.getTensorParallelDegree();
            int ppDegree = pyEnv.getPipelineParallelDegree();
            int partitions = tpDegree * ppDegree;
            if (partitions == 0) {
                partitions = PyEnv.getDefaultTensorParallelDegree();
                tpDegree = partitions / ppDegree;
                pyEnv.setTensorParallelDegree(tpDegree);
                setProperty("tensor_parallel_degree", String.valueOf(tpDegree));
                setProperty("pipeline_parallel_degree", String.valueOf(ppDegree));
                logger.info(
                        "No tensor parallel degree specified. Defaulting to use all available"
                                + " GPUs.");
            }
            logger.info(
                    "Loading model in MPI mode with world size {} (TP {}, PP {}).",
                    partitions,
                    tpDegree,
                    ppDegree);

            int mpiWorkers = pyEnv.getMpiWorkers();
            if (mpiWorkers <= 0) {
                throw new EngineException(
                        "GPU devices are not enough to run " + partitions + " partitions.");
            }

            if (getProperty("minWorkers") == null && getProperty("gpu.minWorkers") == null) {
                setProperty("minWorkers", String.valueOf(mpiWorkers));
                setProperty("gpu.minWorkers", String.valueOf(mpiWorkers));
            }
            if (getProperty("gpu.maxWorkers") == null) {
                if (getProperty("maxWorkers") == null) {
                    setProperty("maxWorkers", String.valueOf(mpiWorkers));
                }
                setProperty("gpu.maxWorkers", getProperty("maxWorkers"));
            }
            if (mpiWorkers < intProperty("gpu.maxWorkers", -1)) {
                throw new IllegalArgumentException(
                        "We can only expand worker to "
                                + mpiWorkers
                                + " but the value is set to "
                                + getProperty("gpu.maxWorkers"));
            }
            mpiWorkers = intProperty("gpu.maxWorkers", -1);

            properties.forEach(pyEnv::addParameter);

            createAllPyProcesses(mpiWorkers, partitions);
        } else {
            int tensorParallelDegree = pyEnv.getTensorParallelDegree();
            if (tensorParallelDegree > 0) {
                if (getProperty("maxWorkers") == null && getProperty("gpu.maxWorkers") == null) {
                    setProperty("gpu.minWorkers", "1");
                    setProperty("gpu.maxWorkers", "1");
                }
                setProperty("tensor_parallel_degree", String.valueOf(tensorParallelDegree));
            }

            properties.forEach(pyEnv::addParameter);
        }
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Device device) {
        int timeout = pyEnv.getPredictTimeout();
        if (pyEnv.isMpiMode()) {
            if (workerQueue.isEmpty()) {
                throw new EngineException("There are no devices left to create new workers");
            }
            return new PyPredictor<>(this, workerQueue.poll(), timeout, translator, device);
        }
        PyProcess worker = new PyProcess(this, pyEnv, -1);
        worker.startPythonProcess();
        return new PyPredictor<>(this, worker, timeout, translator, device);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        shutdown();
    }

    private Path findModelFile(String prefix) {
        if (Files.isRegularFile(modelDir)) {
            Path file = modelDir;
            modelDir = modelDir.getParent();
            if (file.toString().endsWith(".py")) {
                return file;
            }
        } else if (Files.isRegularFile(modelDir.resolve("MAR-INF/MANIFEST.json"))) {
            return Paths.get("");
        }
        if (prefix == null) {
            prefix = modelName;
        }
        Path modelFile = modelDir.resolve(prefix);
        if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
            if (prefix.endsWith(".py")) {
                return null;
            }
            modelFile = modelDir.resolve("model.py");
            if (Files.notExists(modelFile) || !Files.isRegularFile(modelFile)) {
                return null;
            }
        }
        return modelFile;
    }

    private boolean hasModelFile(Path modelDir, String prefix, String... extensions) {
        for (String extension : extensions) {
            if (Files.isRegularFile(modelDir.resolve(prefix + extension))) {
                return true;
            }
            if (Files.isRegularFile(modelDir.resolve("model" + extension))) {
                return true;
            }
        }
        return false;
    }

    private void createAllPyProcesses(int mpiWorkers, int worldSize) {
        long begin = System.currentTimeMillis();
        ExecutorService pool = null;
        List<Future<?>> futures = new ArrayList<>();
        if (parallelLoading) {
            pool = Executors.newFixedThreadPool(mpiWorkers);
        }
        logger.info("Start {} mpiWorkers ...", mpiWorkers);
        int deviceId = manager.getDevice().getDeviceId();
        for (int i = 0; i < mpiWorkers; ++i) {
            logger.debug("Pre-creating python worker: {} ", i);
            PyProcess worker = new PyProcess(this, pyEnv, deviceId + i * worldSize);
            workerQueue.offer(worker);
            if (pool != null) {
                logger.debug("Submitting to pool: {}", i);
                futures.add(pool.submit(worker::startPythonProcess));
            } else {
                worker.startPythonProcess();
            }
        }
        if (pool != null) {
            pool.shutdown();
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    shutdown();
                    throw new EngineException("Failed to start worker", e.getCause()); // NOPMD
                } catch (InterruptedException e) {
                    shutdown();
                    throw new AssertionError("Worker startup interrupted.", e);
                }
            }
        }
        long duration = System.currentTimeMillis() - begin;
        logger.info("{} model loaded in {} ms.", modelName, duration);
    }

    private void shutdown() {
        for (PyProcess process : workerQueue) {
            process.stopPythonProcess(false);
        }
        workerQueue.clear();
        if (pyEnv.isEnableVenv()) {
            pyEnv.deleteVirtualEnv(Utils.hash(modelDir.toString()));
        }
    }
}
