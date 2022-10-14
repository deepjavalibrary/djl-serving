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
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingDeque;
import java.util.stream.Stream;

/** {@code PyModel} is the Python engine implementation of {@link Model}. */
public class PyModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(PyModel.class);

    private PyEnv pyEnv;
    private boolean parallelLoading;
    private LinkedBlockingDeque<PyProcess> workerQueue;

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
        pyEnv = new PyEnv(!"Python".equals(manager.getEngine().getEngineName()));
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
        if (options != null) {
            for (Map.Entry<String, ?> entry : options.entrySet()) {
                String key = entry.getKey();
                String value = (String) entry.getValue();
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
                    case "handler":
                        pyEnv.setHandler(value);
                        break;
                    default:
                        pyEnv.addParameter(key, value);
                        break;
                }
            }
        }

        if (entryPoint == null) {
            entryPoint = Utils.getenv("DJL_ENTRY_POINT");
            if (entryPoint == null) {
                Path modelFile = findModelFile(prefix);
                if (modelFile == null) {
                    throw new FileNotFoundException(".py file not found in: " + modelPath);
                }
                entryPoint = modelFile.toFile().getName();
            }
        }
        pyEnv.setEntryPoint(entryPoint);
        if (pyEnv.isMpiMode()) {
            int partitions;
            if (System.getenv("TENSOR_PARALLEL_DEGREE") != null) {
                partitions = Integer.parseInt(System.getenv("TENSOR_PARALLEL_DEGREE"));
            } else {
                // TODO: avoid use hardcoded "partitioned_model_" name
                try (Stream<Path> stream = Files.list(modelPath)) {
                    partitions =
                            (int)
                                    stream.filter(
                                                    p ->
                                                            p.toFile()
                                                                    .getName()
                                                                    .startsWith(
                                                                            "partitioned_model_"))
                                            .count();
                }
                if (partitions == 0) {
                    throw new FileNotFoundException(
                            "partitioned_model_ file not found in: " + modelPath);
                }
            }
            logger.info("Loading model in MPI model with TP: {}.", partitions);

            pyEnv.setTensorParallelDegree(partitions);
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
            if (mpiWorkers < Integer.parseInt(getProperty("gpu.maxWorkers"))) {
                throw new IllegalArgumentException(
                        "We can only expand worker to "
                                + mpiWorkers
                                + " but the value is set to "
                                + getProperty("gpu.maxWorkers"));
            }
            mpiWorkers = Integer.parseInt(getProperty("gpu.maxWorkers"));
            createAllPyProcesses(mpiWorkers);
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

    private void createAllPyProcesses(int mpiWorkers) {
        ExecutorService pool = null;
        List<Future<?>> futures = new ArrayList<>();
        if (parallelLoading) {
            pool = Executors.newFixedThreadPool(mpiWorkers);
        }
        for (int i = 0; i < mpiWorkers; ++i) {
            logger.debug("Pre-creating python worker: {} ", i);
            PyProcess worker = new PyProcess(this, pyEnv, i);
            workerQueue.offer(worker);
            if (pool != null) {
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
    }

    private void shutdown() {
        for (PyProcess process : workerQueue) {
            process.stopPythonProcess();
        }
        workerQueue.clear();
    }
}
