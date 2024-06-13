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
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.JsonObject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** A class for utils related to SageMaker Secure Mode. */
public final class SecureModeUtils {

    // Platform Secure Mode environment variables – these are protected and can only be set by
    // SageMaker platform
    static final String SECURE_MODE_ENV_VAR = "SAGEMAKER_SECURE_MODE";
    static final String TRUSTED_CHANNELS_ENV_VAR = "SAGEMAKER_TRUSTED_CHANNELS";
    static final String UNTRUSTED_CHANNELS_ENV_VAR = "SAGEMAKER_UNTRUSTED_CHANNELS";
    static final String SECURITY_CONTROLS_ENV_VAR = "SAGEMAKER_SECURITY_CONTROLS";

    // Individual security controls names
    static final String REQUIREMENTS_TXT_CONTROL = "DISALLOW_REQUIREMENTS_TXT";
    static final String PICKLE_FILES_CONTROL = "DISALLOW_PICKLE_FILES";
    static final String TRUST_REMOTE_CODE_CONTROL = "DISALLOW_TRUST_REMOTE_CODE";
    static final String CUSTOM_ENTRYPOINT_CONTROL = "DISALLOW_CUSTOM_ENTRYPOINT";
    static final String CHAT_TEMPLATE_CONTROL = "DISALLOW_CHAT_TEMPLATE";

    private static final String PICKLE_EXTENSIONS_REGEX = ".*\\.(?i:bin|pt|pth|ckpt|pkl)$";

    private static final Logger logger = LoggerFactory.getLogger(SecureModeUtils.class);

    private SecureModeUtils() {}

    /**
     * Check if Secure Mode is enabled via environment variable, and check that associated
     * environment variables are set.
     *
     * @return true if secure mode is enabled, false otherwise
     */
    public static boolean isSecureMode() {
        if (Boolean.parseBoolean(Utils.getenv(SECURE_MODE_ENV_VAR))) {
            if (Utils.getenv(TRUSTED_CHANNELS_ENV_VAR) == null) {
                throw new IllegalArgumentException(
                        "Trusted Channels environment variable is not set.");
            }
            if (Utils.getenv(UNTRUSTED_CHANNELS_ENV_VAR) == null) {
                throw new IllegalArgumentException(
                        "Untrusted Channels environment variable is not set.");
            }
            if (Utils.getenv(SECURITY_CONTROLS_ENV_VAR) == null) {
                throw new IllegalArgumentException(
                        "Security Controls environment variable is not set.");
            }
            logger.info(
                    "Secure Mode is enabled with the following security controls set : {}",
                    Utils.getenv(SECURITY_CONTROLS_ENV_VAR));
            return true;
        }
        return false;
    }

    /**
     * Runs enabled security checks on untrusted paths.
     *
     * @throws IOException if there is an error scanning the paths
     * @throws ModelException if a security check fails
     */
    public static void validateSecurity() throws IOException, ModelException {
        checkOptionEnvVars();
        List<String> untrustedPathList =
                splitCommaSeparatedString(Utils.getenv(UNTRUSTED_CHANNELS_ENV_VAR));
        scanPaths(untrustedPathList);
    }

    /**
     * Handles files from additional model data sources. Currently, this only consists of installing
     * all trusted requirements.txt files. More functionality can be added as needed.
     *
     * @param modelUri the main model directory
     * @throws IOException if there is an error scanning the paths
     * @throws URISyntaxException if modelUri is invalid
     */
    public static void reconcileSources(String modelUri) throws IOException, URISyntaxException {
        List<String> trustedPathList =
                splitCommaSeparatedString(Utils.getenv(TRUSTED_CHANNELS_ENV_VAR));
        Path modelDir = Paths.get(new URI(modelUri));
        linkAdditionalRequirementsTxt(trustedPathList, modelDir);
    }

    /**
     * Check if a disallowed option is set via environment variables. The disallowed value is
     * handled on a per-option basis.
     *
     * @throws ModelException if a security check fails
     */
    private static void checkOptionEnvVars() throws ModelException {
        List<String> securityControls =
                splitCommaSeparatedString(Utils.getenv(SECURITY_CONTROLS_ENV_VAR));

        if (securityControls.contains(TRUST_REMOTE_CODE_CONTROL)) {
            String optionEnvValue = Utils.getenv("OPTION_TRUST_REMOTE_CODE");
            if (optionEnvValue != null && Boolean.parseBoolean(optionEnvValue.trim())) {
                throw new ModelException(
                        "Setting OPTION_TRUST_REMOTE_CODE to True is prohibited in Secure Mode.");
            }
        }
        if (securityControls.contains(CUSTOM_ENTRYPOINT_CONTROL)) {
            // Check for both env vars that can be used to set entrypoint
            String optionEnvValue = Utils.getenv("OPTION_ENTRYPOINT");
            String djlEnvValue = Utils.getenv("DJL_ENTRY_POINT");
            if ((optionEnvValue != null && optionEnvValue.trim().endsWith(".py"))
                    || (djlEnvValue != null && djlEnvValue.trim().endsWith(".py"))) {
                throw new ModelException(
                        "Setting OPTION_ENTRYPOINT or DJL_ENTRY_POINT to a custom Python script is"
                                + " prohibited in Secure Mode. It must be set to a built-in handler"
                                + " provided by LMI (i.e. starting with `djl_python`).");
            }
        }
    }

    /**
     * Given a list of absolute paths, scan each path with enabled security checks.
     *
     * @param pathList list of absolute paths
     * @throws IOException if there is an error scanning the paths
     * @throws ModelException if a security check fails
     */
    private static void scanPaths(List<String> pathList) throws IOException, ModelException {
        List<String> securityControls =
                splitCommaSeparatedString(Utils.getenv(SECURITY_CONTROLS_ENV_VAR));
        for (String path : pathList) {
            Path p = Paths.get(path.trim());
            if (Files.isDirectory(p)) {
                if (securityControls.contains(REQUIREMENTS_TXT_CONTROL)) {
                    scanForDisallowedFile(p, "requirements.txt");
                }
                if (securityControls.contains(PICKLE_FILES_CONTROL)) {
                    scanForPickle(p);
                }
                if (securityControls.contains(TRUST_REMOTE_CODE_CONTROL)) {
                    scanForDisallowedOption(p, "option.trust_remote_code");
                }
                if (securityControls.contains(CUSTOM_ENTRYPOINT_CONTROL)) {
                    scanForDisallowedOption(p, "option.entryPoint");
                    scanForDisallowedFile(p, "model.py");
                }
                if (securityControls.contains(CHAT_TEMPLATE_CONTROL)) {
                    scanForChatTemplate(p);
                }
            } else {
                throw new IllegalArgumentException("Path " + p + " is not a directory.");
            }
        }
    }

    /**
     * Search for a prohibited file in the directory, and fast-fail if found.
     *
     * @param directory the directory to search for the file
     * @param fileName the name of the file to search for
     * @throws IOException if there is an error walking the directory
     * @throws ModelException if a security check fails
     */
    private static void scanForDisallowedFile(Path directory, String fileName)
            throws IOException, ModelException {
        Path filePath = lookForFile(directory, fileName);
        if (filePath != null) {
            throw new ModelException(
                    "File "
                            + fileName
                            + " found at "
                            + filePath.toString()
                            + ", but is prohibited in Secure Mode.");
        }
    }

    /**
     * Search for any pickle-based file in the directory, and fast-fail if found.
     *
     * @param directory the directory to search for pickle-based files
     * @throws IOException if there is an error walking the directory
     * @throws ModelException if a security check fails
     */
    private static void scanForPickle(Path directory) throws IOException, ModelException {
        Pattern pattern = Pattern.compile(PICKLE_EXTENSIONS_REGEX);

        try (Stream<Path> stream = Files.walk(directory)) {
            boolean pickleFound =
                    stream.anyMatch(
                            path -> {
                                Matcher matcher = pattern.matcher(path.toString());
                                return matcher.matches();
                            });
            if (pickleFound) {
                throw new ModelException(
                        "Pickle-based files found in directory "
                                + directory.toString()
                                + ", but only model files are permitted in Secure Mode.");
            }
        }
    }

    /**
     * Check if a disallowed option is set via a serving.properties file contained in an untrusted
     * directory. The disallowed value is handled on a per-option basis. Fast-fail upon a disallowed
     * option being set.
     *
     * @param directory the directory to search for disallowed options
     * @param option the option to check for
     * @throws IOException if there is an error walking the directory
     * @throws ModelException if a security check fails
     */
    private static void scanForDisallowedOption(Path directory, String option)
            throws IOException, ModelException {
        String servingPropertiesValue = null;
        // Walk directory for a serving.properties file containing option line
        Path servingPropertiesFile = lookForFile(directory, "serving.properties");
        if (servingPropertiesFile != null) {
            List<String> lines = Files.readAllLines(servingPropertiesFile);
            for (String line : lines) {
                if (line.toLowerCase(Locale.ROOT).startsWith(option.toLowerCase())) {
                    servingPropertiesValue = line.split("=")[1].trim();
                }
            }
        }
        // Handle disallowed value on a per-option basis
        if ("option.trust_remote_code".equals(option)) {
            // The disallowed value is boolean "true"
            if (Boolean.parseBoolean(servingPropertiesValue)) {
                throw new ModelException(
                        "Setting option.trust_remote_code to True is prohibited in Secure Mode.");
            }
        } else if ("option.entryPoint".equals(option)) {
            // The disallowed value is a .py file
            if (servingPropertiesValue != null && servingPropertiesValue.endsWith(".py")) {
                throw new ModelException(
                        "Setting option.entryPoint to a custom Python script is prohibited in"
                                + " Secure Mode. It must be set to a handler provided by LMI (i.e."
                                + " starting with `djl_python`).");
            }
        } else {
            throw new IllegalArgumentException("Invalid disallowed option: " + option);
        }
    }

    /**
     * Search for a tokenizer_config.json in the directory. If found, check if Jinja chat_template
     * key is present. If so, fast-fail.
     *
     * @param directory the directory to search for tokenizer_config.json
     * @throws IOException if there is an error walking the directory
     * @throws ModelException if a security check fails
     */
    private static void scanForChatTemplate(Path directory) throws IOException, ModelException {
        Path tokenizerConfig = lookForFile(directory, "tokenizer_config.json");
        if (tokenizerConfig != null) {
            try {
                String content = Files.readString(Paths.get(tokenizerConfig.toString()));
                JsonObject jsonObject = JsonUtils.GSON.fromJson(content, JsonObject.class);
                if (jsonObject.has("chat_template")) {
                    throw new ModelException(
                            "Jinja chat_template field found in "
                                    + tokenizerConfig.toString()
                                    + ", but is prohibited in Secure Mode.");
                }
            } catch (IOException e) {
                throw new IOException("Error reading tokenizer_config.json", e);
            }
        }
    }

    /**
     * Link additional requirements.txt files into requirements.txt in modelDir using -r. This
     * single requirements.txt will be installed during Python engine initialization.
     *
     * @param pathList list of absolute paths
     * @param modelDir path to model_dir
     * @throws IOException if there is an error finding requirements.txt files
     */
    private static void linkAdditionalRequirementsTxt(List<String> pathList, Path modelDir)
            throws IOException {
        // Gather requirements.txt found in trusted paths
        List<String> additionalRequirementsTxts = new ArrayList<>();
        for (String path : pathList) {
            Path p = Paths.get(path.trim());
            if (Files.isDirectory(p) && !Files.isSameFile(p, modelDir)) {
                Path requirementsTxt = lookForFile(p, "requirements.txt");
                if (requirementsTxt != null) {
                    additionalRequirementsTxts.add(requirementsTxt.toString());
                }

            } else {
                throw new IllegalArgumentException("Path " + p + " is not a directory.");
            }
        }
        // Append to or create requirements.txt in modelDir
        Path requirementsTxt = lookForFile(modelDir, "requirements.txt");
        if (requirementsTxt == null) {
            requirementsTxt = Files.createFile(modelDir.resolve("requirements.txt"));
        } else {
            logger.info("Existing requirements.txt found at {}", requirementsTxt);
        }
        for (String additionalRequirementsTxt : additionalRequirementsTxts) {
            Files.write(
                    requirementsTxt,
                    ("-r " + additionalRequirementsTxt + "\n").getBytes(StandardCharsets.UTF_8),
                    StandardOpenOption.APPEND);
        }
    }

    /**
     * Walk a directory for a file with the given name.
     *
     * @param directory directory to walk
     * @param fileName name of the file to look for
     * @return the path to the file if found, null otherwise
     * @throws IOException if there is an error walking the directory
     */
    private static Path lookForFile(Path directory, String fileName) throws IOException {
        try (Stream<Path> stream = Files.walk(directory)) {
            Path result =
                    stream.filter(Files::isRegularFile)
                            .filter(p -> p.getFileName().toString().equals(fileName))
                            .findFirst()
                            .orElse(null);
            if (result != null) {
                logger.debug("File {} found at path {}", fileName, result);
            } else {
                logger.debug("File {} not found in {}", fileName, directory);
            }
            return result;
        }
    }

    /**
     * Given an input string, split it into a list of strings using a comma as a delimiter and
     * trimming whitespace.
     *
     * @param input a string containing a comma-separated list of strings
     * @return a list of strings
     */
    private static List<String> splitCommaSeparatedString(String input) {
        if (input == null || input.isEmpty()) {
            return new ArrayList<>();
        }
        return Arrays.stream(input.split(",")).map(String::trim).collect(Collectors.toList());
    }
}
