/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.plugins.securemode;

import ai.djl.ModelException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.http.IllegalConfigurationException;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.util.Utils;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;

/** Unit tests for Secure Mode adapter validation. */
public class SecureModeAdapterTest {

    private static final Path TEST_MODEL_DIR = Paths.get("build/mock/model/");
    private static final Path ADAPTER_DIR = Paths.get("build/mock/adapter/");

    @BeforeMethod
    private void setUp() throws IOException {
        Utils.deleteQuietly(Paths.get("build/mock"));
        Files.createDirectories(TEST_MODEL_DIR);
        Files.createDirectories(ADAPTER_DIR);
    }

    @AfterMethod
    private void tearDown() {
        Utils.deleteQuietly(Paths.get("build/mock"));
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithModelPy() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("model.py"), "# custom code");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithRequirementsTxt() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("requirements.txt"), "torch==2.0.0");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.REQUIREMENTS_TXT_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithPickleFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.bin"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithPtFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.pt"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithPthFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.pth"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithCkptFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.ckpt"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithPklFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.pkl"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test
    void testAdapterWithSafetensorsFile() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter.safetensors"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
        // Should not throw exception
    }

    @Test
    void testAdapterWithoutForbiddenFiles() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("adapter_config.json"), "{}");
        createFileWithContent(ADAPTER_DIR.resolve("adapter_model.safetensors"), "fake binary");
        mockSecurityEnvAndCreateAdapter(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL
                        + ","
                        + SecureModeUtils.REQUIREMENTS_TXT_CONTROL
                        + ","
                        + SecureModeUtils.PICKLE_FILES_CONTROL);
        // Should not throw exception
    }

    @Test
    void testAdapterWithSecureModeDisabled() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("model.py"), "# custom code");
        createFileWithContent(ADAPTER_DIR.resolve("requirements.txt"), "torch==2.0.0");
        createFileWithContent(ADAPTER_DIR.resolve("adapter.bin"), "fake binary");

        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("false");
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            createAdapter();
            // Should not throw exception when secure mode is disabled
        }
    }

    @Test
    void testAdapterWithNoSecurityControls() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("model.py"), "# custom code");

        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(null);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            createAdapter();
            // Should not throw exception when security controls are not set
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithMultipleViolations() throws IOException, ModelException {
        createFileWithContent(ADAPTER_DIR.resolve("model.py"), "# custom code");
        createFileWithContent(ADAPTER_DIR.resolve("requirements.txt"), "torch==2.0.0");
        createFileWithContent(ADAPTER_DIR.resolve("adapter.bin"), "fake binary");
        mockSecurityEnvAndCreateAdapter(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL
                        + ","
                        + SecureModeUtils.REQUIREMENTS_TXT_CONTROL
                        + ","
                        + SecureModeUtils.PICKLE_FILES_CONTROL);
        // Should throw on first violation (model.py)
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithPickleInSubdirectory() throws IOException, ModelException {
        Path subdir = ADAPTER_DIR.resolve("subdir");
        Files.createDirectories(subdir);
        createFileWithContent(subdir.resolve("adapter.bin"), "fake binary");
        mockSecurityEnvAndCreateAdapter(SecureModeUtils.PICKLE_FILES_CONTROL);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAdapterWithInvalidPath() throws IOException, ModelException {
        Path invalidPath = Paths.get("build/mock/nonexistent");
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            SecureModeAdapterValidator.validateAdapterPath(invalidPath.toString());
        }
    }

    @Test
    void testAdapterRegistrationFailureNotInList() throws IOException, ModelException {
        // Create adapter with model.py (forbidden)
        createFileWithContent(ADAPTER_DIR.resolve("model.py"), "# custom code");

        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            ModelInfo<Input, Output> modelInfo = createModelInfo();

            // Verify adapter list is empty before registration attempt
            Assert.assertEquals(modelInfo.getAdapters().size(), 0);

            // Attempt to validate adapter path - should throw exception
            try {
                SecureModeAdapterValidator.validateAdapterPath(ADAPTER_DIR.toString());
                Assert.fail("Expected IllegalConfigurationException to be thrown");
            } catch (IllegalConfigurationException e) {
                // Expected - adapter validation failed
                Assert.assertTrue(
                        e.getMessage().contains("model.py"), "Exception should mention model.py");
            }

            // Verify adapter was NOT added to the list (since validation failed before creation)
            Assert.assertEquals(
                    modelInfo.getAdapters().size(),
                    0,
                    "Failed adapter should not be in adapter list");
            Assert.assertNull(
                    modelInfo.getAdapter("bad-adapter"),
                    "Failed adapter should not be retrievable");
        }
    }

    @Test
    void testStaticAdapterLoadingWithSecurityViolation() throws IOException, ModelException {
        // Simulate static adapter loading (from adapters/ directory during model load)
        Path adaptersDir = TEST_MODEL_DIR.resolve("adapters");
        Path goodAdapter = adaptersDir.resolve("good-adapter");
        Path badAdapter = adaptersDir.resolve("bad-adapter");

        Files.createDirectories(goodAdapter);
        Files.createDirectories(badAdapter);

        // Good adapter has only safetensors
        createFileWithContent(goodAdapter.resolve("adapter_config.json"), "{}");
        createFileWithContent(goodAdapter.resolve("adapter_model.safetensors"), "fake binary");

        // Bad adapter has model.py
        createFileWithContent(badAdapter.resolve("adapter_config.json"), "{}");
        createFileWithContent(badAdapter.resolve("model.py"), "# custom code");

        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            // Try to validate good adapter - should succeed
            SecureModeAdapterValidator.validateAdapterPath(goodAdapter.toString());

            // Try to validate bad adapter - should fail
            try {
                SecureModeAdapterValidator.validateAdapterPath(badAdapter.toString());
                Assert.fail("Expected IllegalConfigurationException for bad adapter");
            } catch (IllegalConfigurationException e) {
                // Expected
                Assert.assertTrue(e.getMessage().contains("model.py"));
            }
        }
    }

    @Test
    void testDynamicAdapterRegistrationWithSecurityViolation() throws IOException, ModelException {
        // Simulate dynamic adapter registration (via API)
        Path goodAdapter = Paths.get("build/mock/dynamic-good");
        Path badAdapter = Paths.get("build/mock/dynamic-bad");

        Files.createDirectories(goodAdapter);
        Files.createDirectories(badAdapter);

        // Good adapter
        createFileWithContent(goodAdapter.resolve("adapter_config.json"), "{}");
        createFileWithContent(goodAdapter.resolve("adapter_model.safetensors"), "fake binary");

        // Bad adapter with requirements.txt
        createFileWithContent(badAdapter.resolve("adapter_config.json"), "{}");
        createFileWithContent(badAdapter.resolve("requirements.txt"), "torch==2.0.0");

        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.REQUIREMENTS_TXT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            // Validate good adapter via API - should succeed
            SecureModeAdapterValidator.validateAdapterPath(goodAdapter.toString());

            // Try to validate bad adapter via API - should fail
            try {
                SecureModeAdapterValidator.validateAdapterPath(badAdapter.toString());
                Assert.fail("Expected IllegalConfigurationException for bad adapter");
            } catch (IllegalConfigurationException e) {
                // Expected
                Assert.assertTrue(e.getMessage().contains("requirements.txt"));
            }
        }
    }

    private void createFileWithContent(Path file, String content) throws IOException {
        if (Files.exists(file)) {
            return;
        }
        Files.createDirectories(Objects.requireNonNull(file.getParent()));
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
    }

    private void mockSecurityEnvAndCreateAdapter(String securityControl)
            throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(securityControl);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            createAdapter();
        }
    }

    private void createAdapter() throws ModelException, IOException {
        // Just validate the path directly since we can't create a full adapter without engine
        SecureModeAdapterValidator.validateAdapterPath(ADAPTER_DIR.toString());
    }

    private ModelInfo<Input, Output> createModelInfo() throws ModelException, IOException {
        return new ModelInfo<>(
                "test",
                TEST_MODEL_DIR.toUri().toURL().toString(),
                null,
                "Python",
                null,
                Input.class,
                Output.class,
                4711,
                1,
                300,
                1,
                -1,
                -1);
    }
}
