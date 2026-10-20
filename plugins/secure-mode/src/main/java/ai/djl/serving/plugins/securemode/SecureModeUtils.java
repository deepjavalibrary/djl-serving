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
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Objects;
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

    private static final Pattern BUILTIN_ENTRY_POINT =
            Pattern.compile("djl_python(\\.[A-Za-z_][A-Za-z0-9_]*)+");

    // What import_module will load: source, sourceless bytecode, or a native extension.
    private static final String[] IMPORTABLE_SUFFIXES = {".py", ".pyc", ".so", ".pyd"};

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
    private static void checkOptions(ModelInfo<?, ?> modelInfo, Set<String> securityControls)
            throws IOException {
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
            if (entryPoint != null && !isBuiltInEntryPoint(entryPoint)) {
                throw new IllegalConfigurationException(
                        "Custom entrypoint is prohibited in Secure Mode.");
            }
            // Checked even when a built-in entry point is set: the Python worker imports model.py
            // at startup whatever the entry point is.
            checkModelPy(modelDir, "model directory");
            checkTorchServeManifest(modelDir, "model directory");
            // Both checked even when no entry point is set, because the engine then selects one.
            checkSelfNamedModule(modelDir, "model directory");
            checkSelectableModule(modelDir, "model directory");
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
     * Returns whether an entry point names a module of the bundled {@code djl_python} package.
     *
     * <p>The prefix alone is not enough. {@code service_loader.load_model_service} treats any entry
     * point ending in {@code .py} as a file to load from the model directory, so {@code
     * djl_python.evil.py} would run a script out of an untrusted model while carrying the expected
     * prefix.
     *
     * @param entryPoint the configured entry point
     * @return true if the entry point is a built-in module rather than a script
     */
    private static boolean isBuiltInEntryPoint(String entryPoint) {
        return !entryPoint.endsWith(".py") && BUILTIN_ENTRY_POINT.matcher(entryPoint).matches();
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
     * Refuses a file whose name is the model directory's own name.
     *
     * <p>{@code PyModel.findModelFile} returns {@code modelDir.resolve(prefix)} whenever that is a
     * regular file, with no extension required, and its name becomes the entry point. The prefix
     * defaults to the directory's name, which an artifact author chooses, so a directory {@code
     * evil.py} holding a file {@code evil.py} is loaded and executed as a script.
     *
     * @param dir the directory to check
     * @param pathDescription description of the path for error messages
     * @throws IllegalConfigurationException if such a file is found
     */
    static void checkSelfNamedModule(Path dir, String pathDescription) {
        Path name = dir.toAbsolutePath().normalize().getFileName();
        if (name == null) {
            return;
        }
        if (Files.isRegularFile(dir.resolve(name.toString()))) {
            throw new IllegalConfigurationException(
                    "Custom inference script "
                            + name
                            + " found in "
                            + pathDescription
                            + ", but custom inference scripts are prohibited in Secure Mode.");
        }
    }

    /**
     * Refuses an extension-less file that sits beside an importable sibling of the same name.
     *
     * <p>With no entry point configured the engine selects one from the model's prefix, and {@code
     * service_loader} imports the name it selects, whichever suffix Python resolves.
     *
     * @param dir the directory to check
     * @param pathDescription description of the path for error messages
     * @throws IOException if the directory cannot be listed
     */
    static void checkSelectableModule(Path dir, String pathDescription) throws IOException {
        if (!Files.isDirectory(dir)) {
            return;
        }
        try (DirectoryStream<Path> files = Files.newDirectoryStream(dir)) {
            for (Path file : files) {
                if (!Files.isRegularFile(file)) {
                    continue;
                }
                String name = Objects.requireNonNull(file.getFileName()).toString();
                String module = importableModuleName(name);
                if (module != null && Files.isRegularFile(dir.resolve(module))) {
                    throw new IllegalConfigurationException(
                            "Custom inference module "
                                    + name
                                    + " found in "
                                    + pathDescription
                                    + ", but custom inference scripts are prohibited in Secure"
                                    + " Mode.");
                }
            }
        }
    }

    /**
     * Returns the module name {@code import_module} would load the file under, or null.
     *
     * <p>An extension module carries an interpreter tag between the module name and the suffix, as
     * in {@code custom.cpython-311-x86_64-linux-gnu.so}, so the name ends at the first dot rather
     * than at the suffix. Matching the suffix exactly would miss every tagged build.
     *
     * @param fileName the file name to interpret
     * @return the module name, or null if Python would not import the file
     */
    private static String importableModuleName(String fileName) {
        for (String suffix : IMPORTABLE_SUFFIXES) {
            if (fileName.endsWith(suffix)) {
                int dot = fileName.indexOf('.');
                return dot > 0 ? fileName.substring(0, dot) : null;
            }
        }
        return null;
    }

    /**
     * Checks if a TorchServe manifest exists in the given directory.
     *
     * <p>{@code service_loader.load_model_service} ignores the entry point when {@code
     * MAR-INF/MANIFEST.json} is present and loads the handler the manifest names instead, so a
     * manifest selects custom code the same way model.py does.
     *
     * @param dir the directory to check
     * @param pathDescription description of the path for error messages
     * @throws IllegalConfigurationException if a manifest is found
     */
    static void checkTorchServeManifest(Path dir, String pathDescription) {
        if (Files.isRegularFile(dir.resolve("MAR-INF/MANIFEST.json"))) {
            throw new IllegalConfigurationException(
                    "TorchServe manifest found in "
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
