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

import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.util.Utils;

import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.net.URL;

public class WorkLoadManagerTest {

    @Test
    public void testFromCriteria() throws IOException {
        WorkLoadManager wlm = new WorkLoadManager();
        String modelUrl = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelUrls(modelUrl)
                        .build();
        ModelInfo<Input, Output> modelInfo = new ModelInfo<>("model", modelUrl, criteria);
        wlm.registerModel(modelInfo).initWorkers(null, 1, 2);
        Input input = new Input();
        URL url = new URL("https://resources.djl.ai/images/0.png");
        try (InputStream is = url.openStream()) {
            input.add(Utils.toByteArray(is));
        }
        Output output = wlm.runJob(new Job<>(modelInfo, input)).join();

        Classifications classification = (Classifications) output.get(0);
        assertEquals(classification.best().getClassName(), "0");
        modelInfo.close();
        wlm.close();
    }
}
