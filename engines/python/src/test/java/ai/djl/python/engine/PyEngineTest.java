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
import ai.djl.engine.Engine;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.DownloadUtils;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import com.google.gson.reflect.TypeToken;
import java.io.IOException;
import java.lang.reflect.Type;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

public class PyEngineTest {

    @Test
    public void testPyEngine() {
        Engine engine = Engine.getInstance();
        Assert.assertNotNull(engine.getVersion());
        Assert.assertTrue(engine.toString().startsWith("Python:"));
        Assert.assertThrows(UnsupportedOperationException.class, () -> engine.setRandomSeed(1));
        Assert.assertThrows(UnsupportedOperationException.class, engine::newGradientCollector);
        Assert.assertThrows(UnsupportedOperationException.class, () -> engine.newSymbolBlock(null));
    }

    @Test
    public void testNDArray() {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray zeros = manager.zeros(new Shape(1, 2));
            float[] data = zeros.toFloatArray();
            Assert.assertEquals(data[0], 0);

            NDArray ones = manager.ones(new Shape(1, 2));
            data = ones.toFloatArray();
            Assert.assertEquals(data[0], 1);

            float[] buf = {0f, 1f, 2f, 3f};
            NDArray array = manager.create(buf);
            Assert.assertEquals(array.toFloatArray(), buf);
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
                        .optEngine("Python")
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

    @Test(enabled = false)
    public void testResnet18() throws TranslateException, IOException, ModelException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("src/test/resources/resnet18"))
                        .optEngine("Python")
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
}
