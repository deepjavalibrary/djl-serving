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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopServingTranslatorFactory;
import ai.djl.translate.TranslateException;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Paths;

public class TsEngineTest {

    @Test(enabled = false)
    public void testTritonModel() throws ModelException, IOException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("/opt/ml/model/simple"))
                        .optTranslatorFactory(new NoopServingTranslatorFactory())
                        .optOption("model_loading_timeout", "1")
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            NDArray input0 = manager.zeros(new Shape(1, 16), DataType.INT32);
            input0.setName("INPUT0");
            NDArray input1 = manager.ones(new Shape(1, 16), DataType.INT32);
            input1.setName("INPUT1");
            NDList list = new NDList(input0, input1);
            Input input = new Input();
            input.add(list);
            input.addProperty("Content-Type", "tensor/ndlist");
            Output output = predictor.predict(input);
            Assert.assertEquals(output.getProperty("Content-Type", null), "tensor/ndlist");
            NDList ret = output.getAsNDList(manager, 0);
            Assert.assertEquals(ret.size(), 2);
            Assert.assertEquals(ret.head().getShape(), new Shape(1, 16));
        }
    }

    @Test(enabled = false)
    public void testNDListInput() throws ModelException, IOException, TranslateException {
        Criteria<NDList, NDList> criteria =
                Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(Paths.get("/opt/ml/model/simple"))
                        .optOption("model_loading_timeout", "1")
                        .optEngine("TritonServer")
                        .build();

        try (ZooModel<NDList, NDList> model = criteria.loadModel();
                Predictor<NDList, NDList> predictor = model.newPredictor();
                NDManager manager = NDManager.newBaseManager()) {
            NDArray input0 = manager.zeros(new Shape(1, 16), DataType.INT32);
            input0.setName("INPUT0");
            NDArray input1 = manager.ones(new Shape(1, 16), DataType.INT32);
            input1.setName("INPUT1");
            NDList list = new NDList(input0, input1);
            NDList ret = predictor.predict(list);
            Assert.assertEquals(ret.size(), 2);
            Assert.assertEquals(ret.head().getShape(), new Shape(1, 16));
        }
    }
}
