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
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.Translator;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileNotFoundException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;

/** {@code PyModel} is the Python engine implementation of {@link Model}. */
public class PyModel extends BaseModel {

    private static final Logger logger = LoggerFactory.getLogger(PyModel.class);

    private PyEnv pyEnv;

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
        pyEnv = new PyEnv();
        dataType = DataType.FLOAT32;
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options)
            throws FileNotFoundException {
        setModelDir(modelPath);
        if (block != null) {
            throw new UnsupportedOperationException(
                    "Python engine does not support dynamic blocks");
        }
        Path modelFile = findModelFile(prefix);
        if (modelFile == null) {
            throw new FileNotFoundException(".py file not found in: " + modelPath);
        }
        pyEnv.setEntryPoint(modelFile.toFile().getName());
        if (options != null) {
            String pythonExecutable = (String) options.get("pythonExecutable");
            if (pythonExecutable != null) {
                pyEnv.setPythonExecutable(pythonExecutable);
            }
            String env = (String) options.get("env");
            if (env != null) {
                String[] envs = env.split(",");
                for (String e : envs) {
                    String[] kv = e.split("=", 2);
                    if (kv.length > 1) {
                        pyEnv.addEnv(kv[0].trim(), kv[1].trim());
                    }
                }
            }
            String predictTimeout = (String) options.get("predict_timeout");
            if (predictTimeout != null) {
                try {
                    int timeoutSeconds = Integer.parseInt(predictTimeout);
                    pyEnv.setPredictTimeout(timeoutSeconds);
                } catch (NumberFormatException ignore) {
                    logger.warn("Invalid predict_timeout value: " + predictTimeout);
                }
            }
            String modelLoadingTimeout = (String) options.get("model_loading_timeout");
            if (modelLoadingTimeout != null) {
                try {
                    int timeoutSeconds = Integer.parseInt(modelLoadingTimeout);
                    pyEnv.setModelLoadingTimeout(timeoutSeconds);
                } catch (NumberFormatException ignore) {
                    logger.warn("Invalid model_loading_timeout value: " + modelLoadingTimeout);
                }
            }
        }
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Device device) {
        return new PyPredictor<>(this, translator, pyEnv, device);
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
}
