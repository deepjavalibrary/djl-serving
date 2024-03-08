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

import static org.testng.Assert.assertEquals;

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
import ai.djl.util.ZipUtils;

import org.testng.Assert;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class ModelInfoTest {

    @BeforeSuite
    public void beforeSuite() throws IOException {
        Path modelStore = Paths.get("build/models");
        Utils.deleteQuietly(modelStore);
        Files.createDirectories(modelStore);
        String engineCacheDir = Utils.getEngineCacheDir().toString();
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        System.setProperty("ENGINE_CACHE_DIR", engineCacheDir);
    }

    @AfterSuite
    public void afterSuite() {
        System.clearProperty("DJL_CACHE_DIR");
        System.clearProperty("ENGINE_CACHE_DIR");
        System.clearProperty("SERVING_FEATURES");
    }

    @Test
    public void testQueueSizeIsSet() {
        ModelInfo<?, ?> modelInfo =
                new ModelInfo<>(
                        "",
                        null,
                        null,
                        "PyTorch",
                        null,
                        Input.class,
                        Output.class,
                        4711,
                        1,
                        300,
                        1,
                        -1,
                        -1);
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
        ModelInfo<Input, Output> modelInfo = new ModelInfo<>("model", modelUrl, criteria);
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
    public void testOutOfMemory() throws IOException, ModelException {
        Path modelDir = Paths.get("build/oom_model");
        Utils.deleteQuietly(modelDir);
        Files.createDirectories(modelDir);

        ModelInfo<Input, Output> modelInfo =
                new ModelInfo<>(
                        "",
                        "build/oom_model",
                        null,
                        "PyTorch",
                        "nc1,nc2",
                        Input.class,
                        Output.class,
                        4711,
                        1,
                        300,
                        1,
                        -1,
                        -1);
        modelInfo.initialize();
        Device device = Engine.getInstance().defaultDevice();
        modelInfo.checkAvailableMemory(device);

        Path file = modelDir.resolve("serving.properties");
        Properties prop = new Properties();
        prop.setProperty("reserved_memory_mb", String.valueOf(Integer.MAX_VALUE));
        prop.setProperty("engine", "PyTorch");
        try (Writer writer = Files.newBufferedWriter(file)) {
            prop.store(writer, "");
        }
        ModelInfo<Input, Output> m1 = new ModelInfo<>("build/oom_model");
        m1.initialize();
        Assert.assertThrows(() -> m1.checkAvailableMemory(Device.cpu()));

        if (device.isGpu()) {
            prop.setProperty("required_memory_mb", "1");
            prop.setProperty("reserved_memory_mb", "1");
            prop.setProperty("gpu.required_memory_mb", String.valueOf(80L * 1024 * 1024 * 1024));
            prop.setProperty("gpu.reserved_memory_mb", "1");
            try (Writer writer = Files.newBufferedWriter(file)) {
                prop.store(writer, "");
            }

            ModelInfo<Input, Output> m2 = new ModelInfo<>("build/oom_model");
            m2.initialize();
            Assert.assertThrows(() -> m2.checkAvailableMemory(device));
        }
    }

    @Test
    public void testInitModel() throws IOException, ModelException {
        Path modelStore = Paths.get("build/models");
        Path modelDir = modelStore.resolve("test_model");
        Files.createDirectories(modelDir);
        Path notModel = modelStore.resolve("non-model");

        ModelInfo<Input, Output> model = new ModelInfo<>(notModel.toUri().toURL().toString());
        Assert.assertThrows(model::initialize);

        model = new ModelInfo<>("build/models/test_model");
        Assert.assertThrows(model::initialize);

        Path xgb = modelDir.resolve("test_model.json");
        Files.createFile(xgb);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "XGBoost");

        Path paddle = modelDir.resolve("__model__");
        Files.createFile(paddle);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "PaddlePaddle");

        Path tflite = modelDir.resolve("test_model.tflite");
        Files.createFile(tflite);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "TFLite");

        Path tensorRt = modelDir.resolve("test_model.uff");
        Files.createFile(tensorRt);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "TensorRT");

        Path onnx = modelDir.resolve("test_model.onnx");
        Files.createFile(onnx);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "OnnxRuntime");

        Path mxnet = modelDir.resolve("test_model-symbol.json");
        Files.createFile(mxnet);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "MXNet");

        Path tensorflow = modelDir.resolve("saved_model.pb");
        Files.createFile(tensorflow);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "TensorFlow");

        Path triton = modelDir.resolve("config.pbtxt");
        Files.createFile(triton);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "TritonServer");

        Path pytorch = modelDir.resolve("test_model.pt");
        Files.createFile(pytorch);
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "PyTorch");

        Path prop = modelDir.resolve("serving.properties");
        try (BufferedWriter writer = Files.newBufferedWriter(prop)) {
            writer.write("engine=MyEngine");
        }
        model = new ModelInfo<>("build/models/test_model");
        model.initialize();
        assertEquals(model.getEngineName(), "MyEngine");

        Path mar = modelStore.resolve("torchServe.mar");
        Path torchServe = modelStore.resolve("torchServe");
        Files.createDirectories(torchServe.resolve("MAR-INF"));
        Files.createDirectories(torchServe.resolve("code"));
        ZipUtils.zip(torchServe, mar, false);
        model = new ModelInfo<>(mar.toUri().toURL().toString());
        model.initialize();
        assertEquals(model.getEngineName(), "Python");

        Path root = modelStore.resolve("models.pt");
        Files.createFile(root);
        model = new ModelInfo<>("build/models");
        model.initialize();
        assertEquals(model.getEngineName(), "PyTorch");
    }

    @Test
    public void testInferLMIEngine() throws IOException, ModelException {
        // vllm/lmi-dist features enabled
        System.setProperty("SERVING_FEATURES", "vllm,lmi-dist");
        Map<String, String> modelToRollingBatch =
                new HashMap<>() {
                    {
                        put("TheBloke/Llama-2-7B-fp16", "lmi-dist");
                        put("openai-community/gpt2", "vllm");
                        put("tiiuae/falcon-7b", "lmi-dist");
                        put("mistralai/Mistral-7B-v0.1", "vllm");
                    }
                };
        Path modelStore = Paths.get("build/models");
        Path modelDir = modelStore.resolve("lmi_test_model");
        Path prop = modelDir.resolve("serving.properties");
        Files.createDirectories(modelDir);
        for (Map.Entry<String, String> entry : modelToRollingBatch.entrySet()) {
            try (BufferedWriter writer = Files.newBufferedWriter(prop)) {
                writer.write("option.model_id=" + entry.getKey());
            }
            ModelInfo<Input, Output> model = new ModelInfo<>("build/models/lmi_test_model");
            model.initialize();
            String inferredRollingBatch = model.getProperties().getProperty("option.rolling_batch");
            assertEquals(inferredRollingBatch, entry.getValue());
            if ("lmi-dist".equals(inferredRollingBatch)) {
                assertEquals(model.getProperties().getProperty("option.mpi_mode"), "true");
            }
        }

        // no features enabled
        System.clearProperty("SERVING_FEATURES");
        try (BufferedWriter writer = Files.newBufferedWriter(prop)) {
            writer.write("option.model_id=tiiuae/falcon-7b");
        }
        ModelInfo<Input, Output> model = new ModelInfo<>("build/models/lmi_test_model");
        model.initialize();
        assertEquals(model.getProperties().getProperty("option.rolling_batch"), "auto");
        assertEquals(model.getProperties().getProperty("option.mpi_mode"), null);

        // invalid hf model case
        try (BufferedWriter writer = Files.newBufferedWriter(prop)) {
            writer.write("option.model_id=invalid-model-id");
        }
        model = new ModelInfo<>("build/models/lmi_test_model");
        Assert.assertThrows(model::initialize);

        // TODO: no good way to test trtllm now since it requires converting the model
    }

    @Test
    public void testInferSageMakerEngine() throws IOException, ModelException {
        ModelInfo<Input, Output> model = new ModelInfo<>("src/test/resources/sagemaker/xgb_model");
        model.initialize();
        assertEquals(model.getEngineName(), "Python");
        assertEquals(model.prop.getProperty("option.entryPoint"), "djl_python.sagemaker");

        // test case for valid pytorch model with customized schema and inference spec
        model = new ModelInfo<>("src/test/resources/sagemaker/pytorch_model");
        model.initialize();
        assertEquals(model.getEngineName(), "Python");
        assertEquals(model.prop.getProperty("option.entryPoint"), "djl_python.sagemaker");

        // test case for invalid model format
        model = new ModelInfo<>("src/test/resources/sagemaker/invalid_pytorch_model");
        Assert.assertThrows(model::initialize);

        // test case for invalid inference spec
        model = new ModelInfo<>("src/test/resources/sagemaker/invalid_inference_spec");
        Assert.assertThrows(model::initialize);
    }
}
