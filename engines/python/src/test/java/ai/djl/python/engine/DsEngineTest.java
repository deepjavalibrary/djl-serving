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

import ai.djl.ModelException;
import ai.djl.engine.EngineException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.cuda.CudaUtils;

import com.google.gson.JsonElement;
import com.google.gson.reflect.TypeToken;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

public class DsEngineTest {

    @BeforeClass
    public void setUp() {
        int gpuCount = CudaUtils.getGpuCount();
        if (gpuCount < 2) {
            throw new SkipException("Large model test requires at least 2 GPUs.");
        }
        if (!System.getProperty("os.name").startsWith("Linux")) {
            throw new SkipException("Large model test only support on Linux.");
        }
        if (!Files.exists(Paths.get("/usr/bin/mpirun"))) {
            throw new SkipException("Large model test requires /usr/bin/mpirun");
        }
        System.setProperty("ENGINE_CACHE_DIR", "build/cache");
    }

    @AfterClass
    public void tierDown() {
        System.clearProperty("ENGINE_CACHE_DIR");
    }

    @Test
    public void testModelLoadingTimeout() {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("src/test/resources/accumulate"))
                        .optTranslator(new NoopTranslator())
                        .optOption("model_loading_timeout", "1")
                        .optEngine("DeepSpeed")
                        .build();

        Assert.assertThrows(EngineException.class, criteria::loadModel);
    }

    @Test
    public void testPredictTimeout() throws IOException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("src/test/resources/accumulate"))
                        .optTranslator(new NoopTranslator())
                        .optOption("predict_timeout", "1")
                        .optEngine("DeepSpeed")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            NDArray x = model.getNDManager().create(new float[] {1});
            Assert.assertThrows(TranslateException.class, () -> predictor.predict(new NDList(x)));
        }
    }

    @Test
    public void testPyModel() throws TranslateException, IOException, ModelException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("src/test/resources/accumulate"))
                        .optTranslator(new NoopTranslator())
                        .optOption("env", "TEST_ENV1=a,TEST_ENV2=b")
                        .optEngine("DeepSpeed")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            NDArray x = model.getNDManager().create(new float[] {1});
            predictor.predict(new NDList(x));
            NDList ret = predictor.predict(new NDList(x));
            float[] expected = {2};
            float[] actual = ret.head().toFloatArray();
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testEchoModel() throws TranslateException, IOException, ModelException {
        // Echo model doesn't support initialize
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("src/test/resources/echo"))
                        .optTranslator(new NoopTranslator())
                        .optEngine("DeepSpeed")
                        .build();
        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor()) {
            NDArray x = model.getNDManager().create(new float[] {1});
            NDList ret = predictor.predict(new NDList(x));
            float[] expected = {1};
            float[] actual = ret.head().toFloatArray();
            Assert.assertEquals(actual, expected);
        }
    }

    @Test
    public void testResnet18() throws TranslateException, IOException, ModelException {
        if (!Boolean.getBoolean("nightly")) {
            return;
        }
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("src/test/resources/resnet18"))
                        .optEngine("DeepSpeed")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            Path file = Paths.get("build/test/kitten.jpg");
            DownloadUtils.download(
                    new URL("https://resources.djl.ai/images/kitten.jpg"), file, null);
            input.add("data", Files.readAllBytes(file));
            input.addProperty("Content-Type", "image/jpeg");
            Output output = predictor.predict(input);
            String classification = output.getData().getAsString();
            Type type = new TypeToken<List<Map<String, Double>>>() {}.getType();
            List<Map<String, Double>> list = JsonUtils.GSON.fromJson(classification, type);
            Assert.assertTrue(list.get(0).containsKey("tabby"));
        }
    }

    @Test
    public void testModelException() throws TranslateException, IOException, ModelException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("src/test/resources/echo"))
                        .optEngine("DeepSpeed")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("exception", "model error");
            Output output = predictor.predict(input);
            Assert.assertEquals(output.getCode(), 424);
            String ret = output.getData().getAsString();
            JsonElement json = JsonUtils.GSON.fromJson(ret, JsonElement.class);
            String error = json.getAsJsonObject().get("error").getAsString();
            Assert.assertEquals(error, "model error");

            // Test empty input
            input = new Input();
            input.add("exception", "");
            output = predictor.predict(input);
            Assert.assertEquals(output.getCode(), 424);
        }
    }

    @Test
    public void testRestartProcess() throws IOException, ModelException, InterruptedException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("src/test/resources/echo"))
                        .optEngine("DeepSpeed")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Input input = new Input();
            input.add("exit", "true");
            Assert.assertThrows(TranslateException.class, () -> predictor.predict(input));

            Input input2 = new Input();
            input2.add("data", "input");
            Output output = null;
            for (int i = 0; i < 5; ++i) {
                Thread.sleep(1000);
                try {
                    output = predictor.predict(input2);
                    break;
                } catch (TranslateException ignore) {
                    // ignore
                }
            }
            Assert.assertNotNull(output);
            Assert.assertEquals(output.getCode(), 200);
        }
    }
}
