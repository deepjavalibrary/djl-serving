/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.tritonserver.engine;

import ai.djl.BaseModel;
import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.translate.ArgumentsUtil;
import ai.djl.translate.Translator;

import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

/** {@code PyModel} is the Python engine implementation of {@link Model}. */
public class TsModel extends BaseModel {

    private TRITONSERVER_Server triton;
    private int timeout;
    JniUtils.ModelMetadata metadata;

    /**
     * Constructs a new Model on a given device.
     *
     * @param modelName the model name
     * @param manager the {@link NDManager} to holds the NDArray
     * @param triton the triton server handle
     */
    TsModel(String modelName, NDManager manager, TRITONSERVER_Server triton) {
        super(modelName);
        this.triton = triton;
        this.manager = manager;
        dataType = DataType.FLOAT32;
        manager.setName("Triton");
    }

    /** {@inheritDoc} */
    @Override
    public void load(Path modelPath, String prefix, Map<String, ?> options) throws IOException {
        setModelDir(modelPath);
        if (block != null) {
            throw new UnsupportedOperationException("Triton does not support dynamic blocks");
        }
        if (!Files.exists(modelDir.resolve("config.pbtxt"))) {
            throw new FileNotFoundException("config.pbtxt file not found in: " + modelDir);
        }
        modelName = modelDir.toFile().getName();

        if (options != null) {
            timeout = ArgumentsUtil.intValue(options, "model_loading_timeout", 120) * 1000;
        } else {
            timeout = 120000;
        }
        JniUtils.loadModel(triton, modelName, timeout);
        wasLoaded = true;
        metadata = JniUtils.getModelMetadata(triton, modelName);
    }

    /** {@inheritDoc} */
    @Override
    public <I, O> Predictor<I, O> newPredictor(Translator<I, O> translator, Device device) {
        return new TsPredictor<>(triton, this, translator, device);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        if (wasLoaded) {
            JniUtils.unloadModel(triton, modelName, timeout);
            wasLoaded = false;
        }
        super.close();
    }
}
