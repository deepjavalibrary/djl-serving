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
import ai.djl.util.Utils;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;

/** Unit tests for the Secure Mode Plugin. */
public class SecureModePluginTest {

    private static final String TEST_FILES_DIR = "/tmp/securemodetest/";
    private static final String TRUSTED_DIR = TEST_FILES_DIR + "trusted/";
    private static final String UNTRUSTED_DIR = TEST_FILES_DIR + "untrusted/";

    // TODO: Refactor tests to improve organization and readability
    // TODO: Add tests for reconcileSources

    @BeforeMethod
    private void setUp() {
        deleteTestFiles(new File(TEST_FILES_DIR));
    }

    @AfterMethod
    private void tearDown() {
        deleteTestFiles(new File(TEST_FILES_DIR));
    }

    private void deleteTestFiles(File directory) {
        if (directory.exists()) {
            File[] files = directory.listFiles();
            if (files != null) {
                for (File file : files) {
                    file.delete();
                }
            }
            directory.delete();
        }
    }

    private void createFileWithContent(String fileName, String content) throws IOException {
        File file = new File(fileName);
        if (!file.getParentFile().exists()) {
            file.getParentFile().mkdirs();
        }
        Files.write(file.toPath(), content.getBytes());
    }

    @Test
    void testSecureModeEnabled() throws IOException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(TRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn("foo");

            boolean result = SecureModeUtils.isSecureMode();
            Assert.assertTrue(result);
        }
    }

    @Test
    void testSecureModeDisabled() throws IOException {
        boolean result = SecureModeUtils.isSecureMode();
        Assert.assertFalse(result);
    }

    @Test(expectedExceptions = IllegalArgumentException.class)
    void testMissingRequiredEnvVar() throws IOException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn("foo");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn("bar");

            boolean result = SecureModeUtils.isSecureMode();
            // expect exception
        }
    }

    private void mockSecurityEnv(String securityControl, String filePath, String fileContent)
            throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(TRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(securityControl);

            createFileWithContent(filePath, fileContent);
            SecureModeUtils.validateSecurity();
        }
    }

    // Untrusted scenarios

    @Test(expectedExceptions = ModelException.class)
    void testUntrustedPickle() throws IOException, ModelException {
        mockSecurityEnv(SecureModeUtils.PICKLE_FILES_CONTROL, UNTRUSTED_DIR + "pickle.bin", "foo");
    }

    @Test(expectedExceptions = ModelException.class)
    void testUntrustedRequirementsTxt() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.REQUIREMENTS_TXT_CONTROL,
                UNTRUSTED_DIR + "requirements.txt",
                "foo");
    }

    @Test(expectedExceptions = ModelException.class)
    void testUntrustedTokenizerChatTemplate() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CHAT_TEMPLATE_CONTROL,
                UNTRUSTED_DIR + "tokenizer_config.json",
                "{\"chat_template\": \"foo\"}");
    }

    @Test
    void testUntrustedTokenizer() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CHAT_TEMPLATE_CONTROL,
                UNTRUSTED_DIR + "tokenizer_config.json",
                "{\"foo\": \"bar\"}");
    }

    @Test(expectedExceptions = ModelException.class)
    void testUntrustedEntryPointPy() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                UNTRUSTED_DIR + "serving.properties",
                "option.entryPoint=model.py");
    }

    @Test
    void testUntrustedEntryPointDJL() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                UNTRUSTED_DIR + "serving.properties",
                "option.entryPoint=djl_python.huggingface");
    }

    @Test(expectedExceptions = ModelException.class)
    void testUntrustedTrustRemoteCodeTrue() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                UNTRUSTED_DIR + "serving.properties",
                "option.trust_remote_code=true");
    }

    @Test
    void testUntrustedTrustRemoteCodeFalse() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                UNTRUSTED_DIR + "serving.properties",
                "option.trust_remote_code=false");
    }

    @Test(expectedExceptions = ModelException.class)
    void testTrustRemoteCodeEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(TRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.TRUST_REMOTE_CODE_CONTROL);
            mockedUtils.when(() -> Utils.getenv("OPTION_TRUST_REMOTE_CODE")).thenReturn("true");

            SecureModeUtils.validateSecurity();
        }
    }

    @Test(expectedExceptions = ModelException.class)
    void testEntrypointOptionEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(TRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.TRUST_REMOTE_CODE_CONTROL);
            mockedUtils.when(() -> Utils.getenv("OPTION_ENTRYPOINT")).thenReturn("model.py");

            SecureModeUtils.validateSecurity();
        }
    }

    @Test(expectedExceptions = ModelException.class)
    void testEntrypointDJLEnvVar() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils = Mockito.mockStatic(Utils.class)) {
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURE_MODE_ENV_VAR))
                    .thenReturn("true");
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.TRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(TRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.UNTRUSTED_CHANNELS_ENV_VAR))
                    .thenReturn(UNTRUSTED_DIR);
            mockedUtils
                    .when(() -> Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR))
                    .thenReturn(SecureModeUtils.TRUST_REMOTE_CODE_CONTROL);
            mockedUtils.when(() -> Utils.getenv("DJL_ENTRY_POINT")).thenReturn("model.py");

            SecureModeUtils.validateSecurity();
        }
    }
}
