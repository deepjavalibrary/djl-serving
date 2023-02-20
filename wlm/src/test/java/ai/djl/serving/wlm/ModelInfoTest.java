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
package ai.djl.serving.wlm;

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

public class ModelInfoTest {

    @Test
    public void testQueueSizeIsSet() {
        ModelInfo<?, ?> modelInfo =
                new ModelInfo<>(
                        "", null, null, "PyTorch", Input.class, Output.class, 4711, 1, 300, 1);
        Assert.assertEquals(4711, modelInfo.getQueueSize());
        Assert.assertEquals(1, modelInfo.getMaxIdleSeconds());
        Assert.assertEquals(300, modelInfo.getMaxBatchDelayMillis());
        Assert.assertEquals(1, modelInfo.getBatchSize());
    }

    @Test
    public void testCriteriaModelInfo() throws ModelException, IOException, TranslateException {
        String modelUrl = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelUrls(modelUrl)
                        .build();
        ModelInfo<Input, Output> modelInfo = new ModelInfo<>("model", criteria);
        modelInfo.load(Device.cpu());
        try (ZooModel<Input, Output> model = modelInfo.getModel(Device.cpu());
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            URL url = new URL("https://resources.djl.ai/images/0.png");
            try (InputStream is = url.openStream()) {
                input.add(Utils.toByteArray(is));
            }
            predictor.predict(input);
        }
    }

    @Test
    public void testOutOfMemory() throws IOException {
        Path modelDir = Paths.get("build/oom_model");
        Utils.deleteQuietly(modelDir);
        Files.createDirectories(modelDir);
        ModelInfo<?, ?> modelInfo =
                new ModelInfo<>(
                        "",
                        "build/oom_model",
                        null,
                        "PyTorch",
                        Input.class,
                        Output.class,
                        4711,
                        1,
                        300,
                        1);

        Device device = Engine.getInstance().defaultDevice();
        modelInfo.checkAvailableMemory(device, modelDir);

        Path file = modelDir.resolve("serving.properties");
        Properties prop = new Properties();
        prop.setProperty("reserved_memory_mb", String.valueOf(Integer.MAX_VALUE));
        try (Writer writer = Files.newBufferedWriter(file)) {
            prop.store(writer, "");
        }
        Assert.assertThrows(() -> modelInfo.checkAvailableMemory(Device.cpu(), modelDir));

        if (device.isGpu()) {
            prop.setProperty("required_memory_mb", "1");
            prop.setProperty("reserved_memory_mb", "1");
            prop.setProperty("gpu.required_memory_mb", String.valueOf(80L * 1024 * 1024 * 1024));
            prop.setProperty("gpu.reserved_memory_mb", "1");
            try (Writer writer = Files.newBufferedWriter(file)) {
                prop.store(writer, "");
            }

            Assert.assertThrows(() -> modelInfo.checkAvailableMemory(device, modelDir));
        }
    }
}
