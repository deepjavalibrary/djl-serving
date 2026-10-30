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
import org.testng.annotations.DataProvider;
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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

    @Test
    void testMissingUntrustedChannels() throws IOException, ModelException {
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
    void testPySuffixEntryPointIsNotABuiltIn() throws IOException, ModelException {
        // The djl_python prefix is not enough: the Python worker loads any entry point ending in
        // .py as a file from the model directory. No model.py here, so this can only throw from
        // the entry-point check.
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.evil.py");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTraversalEntryPointIsNotABuiltIn() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.../../../tmp/evil.py");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUnlistedBuiltInPackageEntryPoint() throws IOException, ModelException {
        // Only the bundled handlers are allowed, not every name under the djl_python package.
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.not_shipped");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAllowlistedEntryPointWithModelPy() throws IOException, ModelException {
        // Naming a built-in entry point does not stop the Python worker importing model.py at
        // startup, so the file is prohibited whichever entry point is set.
        createFileWithContent(TEST_MODEL_DIR.resolve("model.py"), "foo");
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.huggingface");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testAllowlistedEntryPointWithTorchServeManifest() throws IOException, ModelException {
        // service_loader ignores the entry point when a manifest is present and loads the handler
        // the manifest names, so a built-in entry point and no model.py is not enough.
        createFileWithContent(
                TEST_MODEL_DIR.resolve("MAR-INF/MANIFEST.json"),
                "{\"model\":{\"modelName\":\"m\",\"handler\":\"evil.py\"}}");
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.huggingface");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testUrlNamedModuleIsRefused() throws IOException {
        // The model_name argument becomes the prefix, and a dotted name is imported as a package
        // path, so the file it names selects model code.
        createFileWithContent(TEST_MODEL_DIR.resolve("custom.foo"), "");
        String url = TEST_MODEL_DIR.toUri() + "?model_name=custom.foo";
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testEncodedUrlArgumentKeyIsRefused() throws IOException {
        // AbstractRepository decodes the query twice: once via URI.getQuery, then each key and
        // value again. A doubly-encoded key still reaches it as model_name.
        createFileWithContent(TEST_MODEL_DIR.resolve("custom.foo"), "");
        String url = TEST_MODEL_DIR.toUri() + "?model%255Fname=custom.foo";
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testEncodedUrlPathSegmentIsRefused() throws IOException {
        // The path is decoded before the name is taken from its last segment.
        Path script = TEST_MODEL_DIR.resolve("custom.py");
        createFileWithContent(script, "import os");
        String url = TEST_MODEL_DIR.toUri() + "cust%6Fm.py";
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testScriptModelUrlIsRefused() throws IOException {
        // A model URL naming a .py file makes that file the entry point.
        Path script = TEST_MODEL_DIR.resolve("custom.py");
        createFileWithContent(script, "import os");
        String url = script.toUri().toString();
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testFragmentedScriptModelUrlIsRefused() throws IOException {
        // A URI loader resolves the path without the fragment, so the fragment is not part of the
        // name the engine selects either.
        Path script = TEST_MODEL_DIR.resolve("custom.py");
        createFileWithContent(script, "import os");
        String url = script.toUri() + "#fragment";
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTraversalPrefixIsRefused() throws IOException {
        // findModelFile resolves the prefix as given, so one carrying a path still selects a file.
        createFileWithContent(Paths.get("build/mock/evil.py"), "import os");
        String url = TEST_MODEL_DIR.toUri() + "?model_name=../evil.py";
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), TEST_MODEL_DIR);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testNestedSelfNamedModuleIsRefused() throws IOException {
        // The artifact keeps the outer directory's name when the model is its only subdirectory.
        Path outer = Paths.get("build/mock/outer");
        Path inner = outer.resolve("inner");
        createFileWithContent(inner.resolve("outer"), "");
        String url = outer.toUri().toString();
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), inner);
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testSelfNamedModuleIsRefused() throws IOException, ModelException {
        // The model directory is build/mock/model, so with nothing configuring an entry point a
        // file named "model" is what the engine resolves the prefix to.
        createFileWithContent(TEST_MODEL_DIR.resolve("model"), "");
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "engine=Python");
    }

    @Test
    void testSelfNamedModuleAllowedWithBuiltInEntryPoint() throws IOException, ModelException {
        // PyModel calls findModelFile only when entryPoint is still null, so with a built-in one
        // configured the file is never selected and refusing it would be a false positive.
        createFileWithContent(TEST_MODEL_DIR.resolve("model"), "");
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.entryPoint=djl_python.huggingface");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testSelfNamedScriptIsRefused() throws IOException {
        // A .py prefix is selected the same way and loaded as a file rather than imported.
        Path dir = Paths.get("build/mock/evil.py");
        createFileWithContent(dir.resolve("evil.py"), "import os");
        String url = dir.toUri().toString();
        SecureModeUtils.checkSelectedModelFile(new ModelInfo<Input, Output>(url), dir);
    }

    @Test
    void testUnpairedFilesAreAllowed() throws IOException, ModelException {
        // Files the prefix does not name are not selected, whatever siblings they have.
        createFileWithContent(TEST_MODEL_DIR.resolve("custom"), "");
        createFileWithContent(TEST_MODEL_DIR.resolve("custom.py"), "import os");
        createFileWithContent(TEST_MODEL_DIR.resolve("modeling_qwen.py"), "import os");
        // No entryPoint, so the selection check actually runs.
        mockSecurityEnv(
                SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "engine=Python");
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

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTrustRemoteCodeNumericOneIsProhibited() throws IOException, ModelException {
        // pydantic coerces "1" to true, so the worker trusts remote code while
        // Boolean.parseBoolean read it as false and let the model through.
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=1");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTrustRemoteCodeYesIsProhibited() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=YES");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testTrustRemoteCodeMixedCaseTrueIsProhibited() throws IOException, ModelException {
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=TrUe");
    }

    @DataProvider(name = "falsySpellings")
    public Object[][] falsySpellings() {
        return new Object[][] {{"false"}, {"0"}, {"off"}, {"no"}, {"f"}, {"n"}, {"FALSE"}};
    }

    @Test(dataProvider = "falsySpellings")
    void testTrustRemoteCodeFalseSpellingsStillLoad(String value)
            throws IOException, ModelException {
        // The falsy spellings pydantic accepts must keep loading, or the control starts
        // rejecting models that never enabled it. One case per value: the fixture helper skips a
        // file that already exists, so a loop here would re-test whichever value wrote it first.
        mockSecurityEnv(
                SecureModeUtils.TRUST_REMOTE_CODE_CONTROL,
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.trust_remote_code=" + value);
    }

    @Test
    void testIsTruthyMatchesPydanticCoercion() {
        for (String value : new String[] {"1", "true", "on", "yes", "t", "y", "TRUE"}) {
            Assert.assertTrue(SecureModeUtils.isTruthy(value), "should be true: " + value);
        }
        for (String value :
                new String[] {"0", "false", "off", "no", "f", "n", "FALSE", "", "flase", null}) {
            Assert.assertFalse(SecureModeUtils.isTruthy(value), "should be false: " + value);
        }
    }

    @Test
    void testIsTruthyTrimsWherePydanticWouldReject() {
        // Trimming is a deliberate tightening: pydantic does not strip, so " 1 " is a validation
        // error there and prohibited here. Both block; neither lets it through.
        Assert.assertTrue(SecureModeUtils.isTruthy(" 1 "));
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testPropertiesAllowlist() throws IOException, ModelException {
        mockSecurityEnv(
                "foo", TEST_MODEL_DIR.resolve("serving.properties"), "option.not_allowlisted=foo");
    }

    @Test(expectedExceptions = IllegalConfigurationException.class)
    void testInvalidPythonExecutablePath() throws IOException, ModelException {
        mockSecurityEnv(
                "foo",
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.pythonExecutable=/foo/bar/python3");
    }

    @Test
    void testAllowedPythonExecutablePath() throws IOException, ModelException {
        mockSecurityEnv(
                "foo",
                TEST_MODEL_DIR.resolve("serving.properties"),
                "option.pythonExecutable=/opt/djl/vllm_venv/bin/python");
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
        try (MockedStatic<Utils> mockedUtils =
                Mockito.mockStatic(Utils.class, Mockito.CALLS_REAL_METHODS)) {
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
