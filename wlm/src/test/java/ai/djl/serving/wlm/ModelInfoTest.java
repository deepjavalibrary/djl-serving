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
import java.net.URL;

public class ModelInfoTest {

    @Test
    public void testQueueSizeIsSet() {
        try (ModelInfo<?, ?> modelInfo =
                new ModelInfo<>(
                        "", null, null, "MXNet", Input.class, Output.class, 4711, 1, 300, 1)) {
            Assert.assertEquals(4711, modelInfo.getQueueSize());
            Assert.assertEquals(1, modelInfo.getMaxIdleTime());
            Assert.assertEquals(300, modelInfo.getMaxBatchDelay());
            Assert.assertEquals(1, modelInfo.getBatchSize());
        }
    }

    @Test
    public void testCriteriaModelInfo() throws ModelException, IOException, TranslateException {
        String modelUrl = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelUrls(modelUrl)
                        .build();
        try (ModelInfo<Input, Output> modelInfo = new ModelInfo<>(criteria)) {
            modelInfo.load(Device.cpu());
            try (ZooModel<Input, Output> model = modelInfo.getModel(Device.cpu())) {
                try (Predictor<Input, Output> predictor = model.newPredictor()) {
                    Input input = new Input();
                    URL url = new URL("https://resources.djl.ai/images/0.png");
                    try (InputStream is = url.openStream()) {
                        input.add(Utils.toByteArray(is));
                    }
                    predictor.predict(input);
                }
            }
        }
    }
}
