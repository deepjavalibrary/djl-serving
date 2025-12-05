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

import ai.djl.serving.http.IllegalConfigurationException;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.JsonObject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** A class for utils related to SageMaker Secure Mode. */
public final class SecureModeUtils {

    // Platform Secure Mode environment variables – these are protected and can only be set by
    // SageMaker platform
    public static final String SECURE_MODE_ENV_VAR = "SAGEMAKER_SECURE_MODE";
    public static final String UNTRUSTED_CHANNELS_ENV_VAR = "SAGEMAKER_UNTRUSTED_CHANNELS";
    public static final String SECURITY_CONTROLS_ENV_VAR = "SAGEMAKER_SECURITY_CONTROLS";

    // Individual security controls names
    public static final String REQUIREMENTS_TXT_CONTROL = "DISALLOW_REQUIREMENTS_TXT";
    public static final String PICKLE_FILES_CONTROL = "DISALLOW_PICKLE_FILES";
    public static final String TRUST_REMOTE_CODE_CONTROL = "DISALLOW_TRUST_REMOTE_CODE";
    public static final String CUSTOM_ENTRYPOINT_CONTROL = "DISALLOW_CUSTOM_INFERENCE_SCRIPTS";
    public static final String CHAT_TEMPLATE_CONTROL = "DISALLOW_CHAT_TEMPLATE";

    private static final Pattern PICKLE_EXTENSIONS_REGEX =
            Pattern.compile(".*\\.(?i:bin|pt|pth|ckpt|pkl)$");

    private static final Logger logger = LoggerFactory.getLogger(SecureModeUtils.class);

    private SecureModeUtils() {}

    /**
     * Checks if Secure Mode is enabled via environment variable.
     *
     * @return true if secure mode is enabled, false otherwise
     */
    public static boolean isSecureMode() {
        return Boolean.parseBoolean(Utils.getenv(SECURE_MODE_ENV_VAR));
    }

    /**
     * Runs enabled security checks on untrusted paths.
     *
     * @param modelInfo the model
     * @throws IOException if there is an error scanning the paths
     */
    public static void validateSecurity(ModelInfo<?, ?> modelInfo) throws IOException {
        String securityControls = Utils.getenv(SECURITY_CONTROLS_ENV_VAR);
        String untrustedChannels = Utils.getenv(UNTRUSTED_CHANNELS_ENV_VAR);
        if (securityControls == null) {
            throw new IllegalConfigurationException(
                    "Security Controls environment variable is not set.");
        }
        logger.info(
                "Secure Mode enabled with the following security controls: {}", securityControls);

        String engine = modelInfo.getEngineName();
        if (!"Python".equals(engine) && !"MPI".equals(engine)) {
            logger.info("Skipping security check for engine: {}", engine);
            return;
        }

        Set<String> controls = new HashSet<>(Arrays.asList(securityControls.split("\\s*,\\s*")));

        validateProperties(modelInfo, SecureModeAllowList.PROPERTIES_ALLOWLIST);
        checkOptions(modelInfo, controls);
        if (untrustedChannels != null) {
            String[] untrustedPathList = untrustedChannels.split(",");
            scanForbiddenFiles(untrustedPathList, controls);
        }
    }

    /**
     * In Secure Mode we only allow an explicit set of DJL-Serving properties to be set.
     *
     * @param modelInfo ModelInfo of the model
     * @param allowedKeys set of allowlisted properties
     */
    private static void validateProperties(ModelInfo<?, ?> modelInfo, Set<String> allowedKeys) {
        Properties prop = modelInfo.getProperties();
        allowedKeys = new HashSet<>(allowedKeys);
        for (String key : prop.stringPropertyNames()) {
            if (!allowedKeys.contains(key)) {
                throw new IllegalConfigurationException(
                        "Property " + key + " is prohibited from being set in Secure Mode.");
            }
        }
    }

    /**
     * Check if a disallowed option is set via environment variables. The disallowed value is
     * handled on a per-option basis.
     */
    private static void checkOptions(ModelInfo<?, ?> modelInfo, Set<String> securityControls) {
        Properties prop = modelInfo.getProperties();
        Path modelDir = modelInfo.getModelDir();
        if (securityControls.contains(TRUST_REMOTE_CODE_CONTROL)) {
            if (Boolean.parseBoolean(prop.getProperty("option.trust_remote_code"))) {
                throw new IllegalConfigurationException(
                        "Setting TRUST_REMOTE_CODE to True is prohibited in Secure Mode.");
            }
        }
        if (securityControls.contains(CUSTOM_ENTRYPOINT_CONTROL)) {
            String entryPoint =
                    prop.getProperty(
                            "option.entryPoint",
                            Utils.getenv("DJL_ENTRY_POINT", Utils.getenv("OPTION_ENTRYPOINT")));
            if (entryPoint != null) {
                if (!entryPoint.startsWith("djl_python.")) {
                    throw new IllegalConfigurationException(
                            "Custom entrypoint is prohibited in Secure Mode.");
                }
            } else {
                checkModelPy(modelDir, "model directory");
            }
        }
        if (securityControls.contains(REQUIREMENTS_TXT_CONTROL)) {
            checkRequirementsTxt(modelDir, "model directory");
        }
        String pythonExecutable = prop.getProperty("option.pythonExecutable");
        if (pythonExecutable != null
                && !SecureModeAllowList.PYTHON_EXECUTABLE_ALLOWLIST.contains(pythonExecutable)) {
            throw new IllegalConfigurationException(
                    "Custom Python executable path is prohibited in Secure Mode. "
                            + "Only the following paths are allowed: "
                            + SecureModeAllowList.PYTHON_EXECUTABLE_ALLOWLIST);
        }
    }

    /**
     * Checks if model.py exists in the given directory.
     *
     * @param dir the directory to check
     * @param pathDescription description of the path for error messages
     * @throws IllegalConfigurationException if model.py is found
     */
    static void checkModelPy(Path dir, String pathDescription) {
        if (Files.isRegularFile(dir.resolve("model.py"))) {
            throw new IllegalConfigurationException(
                    "Custom model.py found in "
                            + pathDescription
                            + ", but custom inference scripts are prohibited in Secure Mode.");
        }
    }

    /**
     * Checks if requirements.txt exists in the given directory.
     *
     * @param dir the directory to check
     * @param pathDescription description of the path for error messages
     * @throws IllegalConfigurationException if requirements.txt is found
     */
    static void checkRequirementsTxt(Path dir, String pathDescription) {
        if (Files.isRegularFile(dir.resolve("requirements.txt"))) {
            throw new IllegalConfigurationException(
                    "requirements.txt found in "
                            + pathDescription
                            + ", but installing additional dependencies is prohibited in Secure"
                            + " Mode.");
        }
    }

    /**
     * Given a list of absolute paths, scan each path with enabled security checks.
     *
     * @param pathList list of absolute paths
     * @throws IOException if there is an error scanning the paths
     */
    private static void scanForbiddenFiles(String[] pathList, Set<String> securityControls)
            throws IOException {
        for (String path : pathList) {
            Path dir = Paths.get(path.trim());
            if (!Files.isDirectory(dir)) {
                throw new IllegalConfigurationException("Path " + dir + " is not a directory.");
            }

            boolean checkPickle = securityControls.contains(PICKLE_FILES_CONTROL);
            boolean checkJinja = securityControls.contains(CHAT_TEMPLATE_CONTROL);

            if (checkPickle) {
                scanPickleFiles(dir, path);
            }

            if (checkJinja) {
                scanChatTemplates(dir);
            }
        }
    }

    /**
     * Scans a directory for pickle-based files.
     *
     * @param dir the directory to scan
     * @param pathDescription description of the path for error messages
     * @throws IOException if there is an error scanning the directory
     */
    static void scanPickleFiles(Path dir, String pathDescription) throws IOException {
        try (Stream<Path> stream = Files.walk(dir)) {
            for (Path p : stream.collect(Collectors.toList())) {
                Path fileName = p.getFileName();
                if (fileName != null) {
                    String name = fileName.toString();
                    if (PICKLE_EXTENSIONS_REGEX.matcher(name).matches()) {
                        throw new IllegalConfigurationException(
                                "Pickle-based file "
                                        + name
                                        + " found in "
                                        + pathDescription
                                        + ", but only the Safetensors format is permitted in Secure"
                                        + " Mode.");
                    }
                }
            }
        }
    }

    /**
     * Scans a directory for Jinja chat templates in tokenizer_config.json files.
     *
     * @param dir the directory to scan
     * @throws IOException if there is an error scanning the directory
     */
    private static void scanChatTemplates(Path dir) throws IOException {
        try (Stream<Path> stream = Files.walk(dir)) {
            for (Path p : stream.collect(Collectors.toList())) {
                String name = p.toFile().getName();
                if ("tokenizer_config.json".equals(name)) {
                    try (Reader reader = Files.newBufferedReader(p)) {
                        JsonObject jsonObject = JsonUtils.GSON.fromJson(reader, JsonObject.class);
                        if (jsonObject.has("chat_template")) {
                            throw new IllegalConfigurationException(
                                    "Jinja chat_template field found in "
                                            + p
                                            + ", but is prohibited in Secure Mode.");
                        }
                    }
                }
            }
        }
    }
}
