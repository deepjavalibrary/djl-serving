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
import java.util.Map;
import java.util.Objects;

/** Unit tests for the Secure Mode Plugin. */
public class SecureModePluginTest {

    private static final Path TEST_MODEL_DIR = Paths.get("build/mock/model/");
    private static final Path UNTRUSTED_DIR = Paths.get("build/mock/untrusted/");

    @BeforeMethod
    private void setUp() throws IOException {
        Utils.deleteQuietly(Paths.get("build/mock"));
        Files.createDirectories(TEST_MODEL_DIR);
        Files.createDirectories(UNTRUSTED_DIR);
    }

    @AfterMethod
    private void tearDown() {
        Utils.deleteQuietly(Paths.get("build/mock"));
    }

    @Test
    void testSecureModeEnabled() {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            Assert.assertTrue(SecureModeUtils.isSecureMode());
        }
    }

    @Test
    void testSecureModeDisabled() {
        Assert.assertFalse(SecureModeUtils.isSecureMode());
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testMissingSecurityControl() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn("bar");
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            validateSecurity("Python");
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testMissingUntrustedChannels() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn("bar");
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            validateSecurity("Python");
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testInvalidUntrustedChannels() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn("bar");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn("bar");
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            validateSecurity("Python");
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUntrustedPickle() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.PICKLE_FILES_CONTROL, UNTRUSTED_DIR.resolve("pickle.bin"), "foo");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUntrustedRequirementsTxt() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.REQUIREMENTS_TXT_CONTROL,
                TEST_MODEL_DIR.resolve("requirements.txt"),
                "foo");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUntrustedTokenizerChatTemplate() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CHAT_TEMPLATE_CONTROL,
                UNTRUSTED_DIR.resolve("tokenizer_config.json"),
                "{\"chat_template\": \"foo\"}");
    }

    @Test
    void testUntrustedTokenizer() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CHAT_TEMPLATE_CONTROL,
                UNTRUSTED_DIR.resolve("tokenizer_config.json"),
                "{\"foo\": \"bar\"}");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUntrustedEntryPointPyProps() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=model.py");
    }

    @Test
    void testUntrustedEntryPointDJLProps() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.huggingface");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUntrustedEntryPointPyFile() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("model.py"),
                "foo");
    }

    @Test
    void testUntrustedTrustRemoteCodeFalse() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=false");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTrustRemoteCodeTrue() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=true");
    }

    @Test
    void testSkipValidation() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "engine=PyTorch\noption.trust_remote_code=true",
                null);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTrustRemoteCodeEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR.toString());
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.TRUST_REMOTE_CODE_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);
            mockedUtils.when(Utils::getenv).thenReturn(Map.of("OPTION_TRUST_REMOTE_CODE", "true"));

            validateSecurity("Python");
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testEntrypointOptionEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR.toString());
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);
            mockedUtils.when(() -> Utils.getenv("OPTION_ENTRYPOINT")).thenReturn("model.py");
            mockedUtils
                    .when(() -> Utils.getenv("DJL_ENTRY_POINT", "model.py"))
                    .thenReturn("model.py");

            validateSecurity("Python");
        }
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testEntrypointDJLEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR.toString());
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);
            mockedUtils.when(() -> Utils.getenv("DJL_ENTRY_POINT", null)).thenReturn("model.py");

            validateSecurity("Python");
        }
    }

    private void createFileWithContent(Path file, String content) throws IOException {
        if (Files.exists(file)) {
            return;
        }
        Files.createDirectories(Objects.requireNonNull(file.getParent()));
        Files.write(file, content.getBytes(StandardCharsets.UTF_8));
    }

    private void validateSecurity(String engine) throws ModelException, IOException {
        ModelInfo<?, ?> modelInfo =
                new ModelInfo<>(
                        "",
                        TEST_MODEL_DIR.toUri().toURL().toString(),
                        null,
                        engine,
                        null,
                        Input.class,
                        Output.class,
                        4711,
                        1,
                        300,
                        1,
                        -1,
                        -1);
        modelInfo.initialize();
        new SecureModeModelServerListener().onModelConfigured(modelInfo);
    }

    private void mockSecurityEnv(String securityControl, Path file, String fileContent)
            throws ModelException, IOException {
        mockSecurityEnv(securityControl, file, fileContent, "Python");
    }

    private void mockSecurityEnv(
            String securityControl, Path file, String fileContent, String engine)
            throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR.toString());
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(securityControl);
            mockedUtils
                    .when(() -> Utils.getNestedModelDir(Mockito.any()))
                    .thenReturn(TEST_MODEL_DIR);

            createFileWithContent(file, fileContent);
            validateSecurity(engine);
        }
    }
}
