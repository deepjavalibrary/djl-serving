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
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;

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

        // Test exception cases
        Criteria<Input, Output> criteria2 =
                criteria.toBuilder().optOption("pythonExecutable", "non-exists").build();
        Assert.assertThrows(EngineException.class, criteria2::loadModel);

        System.setProperty("DJL_VENV_DIR", "/COM1"); // invalid path
        Assert.assertThrows(EngineException.class, criteria::loadModel);
    }

    @Test
    public void testEngineMetricsSwitch() {
        String name = "SERVING_ENGINE_METRICS";
        if (System.getenv(name) != null) {
            throw new SkipException(name + " is set in the environment.");
        }
        try {
            // off unless it is set to true
            Assert.assertFalse(PyEnv.isEngineMetricsEnabled());
            for (String value : new String[] {"", "false", "0", "1", "yes", "tru"}) {
                System.setProperty(name, value);
                Assert.assertFalse(PyEnv.isEngineMetricsEnabled(), value);
            }
            for (String value : new String[] {"true", "TRUE", "True"}) {
                System.setProperty(name, value);
                Assert.assertTrue(PyEnv.isEngineMetricsEnabled(), value);
            }
        } finally {
            System.clearProperty(name);
        }
    }

    @Test
    public void testModelDirLeavesPythonPathWhenCustomScriptsAreDisallowed() {
        String sep = File.pathSeparator;
        // Absolute for this platform: a leading slash is drive-relative on Windows, which the
        // relative-entry rule drops.
        String extra = Paths.get("extra").toAbsolutePath().toString();
        String cache = Paths.get("cache").toAbsolutePath().toString();
        String model = Paths.get("model").toAbsolutePath().toString();
        // A trailing separator leaves an empty entry and "." is relative; the worker's cwd is the
        // model directory, so either one imports from the model.
        String inherited = extra + sep + sep + "." + sep + "lib";

        List<String> allowed = entries(PyEnv.buildPythonPath(inherited, cache, model, false));
        Assert.assertTrue(allowed.contains(model), allowed.toString());

        List<String> refused = entries(PyEnv.buildPythonPath(inherited, cache, model, true));
        Assert.assertFalse(refused.contains(model), refused.toString());
        Assert.assertFalse(refused.contains(""), refused.toString());
        Assert.assertFalse(refused.contains("."), refused.toString());
        Assert.assertFalse(refused.contains("lib"), refused.toString());
        Assert.assertEquals(refused, Arrays.asList(extra, cache));
    }

    @Test
    public void testCustomScriptControlComesFromTheControlsList() {
        // The control is a value inside SAGEMAKER_SECURITY_CONTROLS, which is how the platform
        // delivers it; an environment that already sets these would mask the assertions.
        String[] names = {"SAGEMAKER_SECURE_MODE", "SAGEMAKER_SECURITY_CONTROLS"};
        for (String name : names) {
            if (System.getenv(name) != null) {
                throw new SkipException(name + " is set in the environment.");
            }
        }
        try {
            Assert.assertFalse(PyEnv.customScriptsDisallowed());

            System.setProperty("SAGEMAKER_SECURITY_CONTROLS", "DISALLOW_CUSTOM_INFERENCE_SCRIPTS");
            Assert.assertFalse(PyEnv.customScriptsDisallowed(), "secure mode is off");

            System.setProperty("SAGEMAKER_SECURE_MODE", "true");
            Assert.assertTrue(PyEnv.customScriptsDisallowed());

            System.setProperty(
                    "SAGEMAKER_SECURITY_CONTROLS",
                    "DISALLOW_PICKLE_FILES, DISALLOW_CUSTOM_INFERENCE_SCRIPTS");
            Assert.assertTrue(PyEnv.customScriptsDisallowed(), "one control of several");

            System.setProperty("SAGEMAKER_SECURITY_CONTROLS", "DISALLOW_PICKLE_FILES");
            Assert.assertFalse(PyEnv.customScriptsDisallowed(), "control not listed");
        } finally {
            for (String name : names) {
                System.clearProperty(name);
            }
        }
    }

    @Test
    public void testSelectedPyFileIsRefusedWhateverNamedIt()
            throws ModelNotFoundException, MalformedModelException, IOException {
        // Checked on the resolved file, so any name that reaches it is refused.
        String[] names = {"SAGEMAKER_SECURE_MODE", "SAGEMAKER_SECURITY_CONTROLS"};
        for (String name : names) {
            if (System.getenv(name) != null) {
                throw new SkipException(name + " is set in the environment.");
            }
        }
        Path dir = Paths.get("build/secure-entrypoint").toAbsolutePath();
        Utils.deleteQuietly(dir);
        Files.createDirectories(dir);
        Path script = dir.resolve("cust%6Fm.py");
        Files.write(script, "x = 1".getBytes(StandardCharsets.UTF_8));

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optEngine("Python")
                        .optModelPath(script)
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel()) {
            Assert.assertNotNull(model, "loads when the control is absent");
        }

        System.setProperty("SAGEMAKER_SECURE_MODE", "true");
        System.setProperty("SAGEMAKER_SECURITY_CONTROLS", "DISALLOW_CUSTOM_INFERENCE_SCRIPTS");
        try {
            Assert.assertThrows(IllegalArgumentException.class, criteria::loadModel);
        } finally {
            for (String name : names) {
                System.clearProperty(name);
            }
        }
    }

    private static List<String> entries(String pythonPath) {
        return Arrays.asList(pythonPath.split(File.pathSeparator, -1));
    }
}
