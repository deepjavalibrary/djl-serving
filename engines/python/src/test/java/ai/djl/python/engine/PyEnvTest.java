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
package ai.djl.python.engine;

import ai.djl.MalformedModelException;
import ai.djl.engine.EngineException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class PyEnvTest {

    @BeforeClass
    public void setUp() {
        System.setProperty("DJL_VENV_DIR", "build/venv");
    }

    @AfterClass
    public void tierDown() {
        System.clearProperty("DJL_VENV_DIR");
        Utils.deleteQuietly(Paths.get("build/venv"));
    }

    @Test
    public void testPythonVenv()
            throws ModelNotFoundException, MalformedModelException, IOException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optEngine("Python")
                        .optModelPath(Paths.get("src/test/resources/echo"))
                        .optOption("enable_venv", "true")
                        .build();

        Path venvDir;
        try (ZooModel<Input, Output> model = criteria.loadModel();
                ZooModel<Input, Output> model2 = criteria.loadModel()) {
            String venvName = Utils.hash(model.getModelPath().toString());
            venvDir = Paths.get("build/venv").resolve(venvName);
            Assert.assertTrue(Files.exists(venvDir));
            Assert.assertNotNull(model2.getModelPath());
        }
        Assert.assertFalse(Files.exists(venvDir));

        // Test exception
        if (!System.getProperty("os.name").startsWith("Win")) {
            System.setProperty("DJL_VENV_DIR", "/non_exist/venv");
            Assert.assertThrows(EngineException.class, criteria::loadModel);
        }
    }
}
