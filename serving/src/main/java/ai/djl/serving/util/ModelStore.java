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
package ai.djl.serving.util;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.FilenameUtils;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.BadWorkflowException;
import ai.djl.serving.workflow.Workflow;
import ai.djl.serving.workflow.WorkflowDefinition;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedWriter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/** A class represent model server's model store. */
public final class ModelStore {

    private static final Logger logger = LoggerFactory.getLogger(ModelStore.class);
    private static final Pattern MODEL_STORE_PATTERN = Pattern.compile("(\\[?([^?]+?)]?=)?(.+)");

    private static final ModelStore INSTANCE = new ModelStore();

    private List<Workflow> workflows;

    private ModelStore() {
        workflows = new ArrayList<>();
    }

    /**
     * Returns the {@code ModelStore} singleton instance.
     *
     * @return the {@code ModelStore} singleton instance
     */
    public static ModelStore getInstance() {
        return INSTANCE;
    }

    /**
     * Initializes the model store.
     *
     * @throws IOException if failed read model from file system
     * @throws BadWorkflowException if failed parse workflow definition
     */
    public void initialize() throws IOException, BadWorkflowException {
        workflows.clear();
        Set<String> startupModels = ModelManager.getInstance().getStartupWorkflows();
        ConfigManager configManager = ConfigManager.getInstance();

        String loadModels = configManager.getLoadModels();
        Path modelStore = configManager.getModelStore();
        if (loadModels == null || loadModels.isEmpty()) {
            loadModels = "ALL";
        }

        Set<String> urls = new HashSet<>();
        if ("NONE".equalsIgnoreCase(loadModels)) {
            // to disable load all models from model store
            return;
        } else if ("ALL".equalsIgnoreCase(loadModels)) {
            if (modelStore == null) {
                logger.warn("Model store is not configured.");
                return;
            }

            if (Files.isDirectory(modelStore)) {
                // contains only directory or archive files
                boolean isMultiModelsDirectory;
                try (Stream<Path> stream = Files.list(modelStore)) {
                    isMultiModelsDirectory = stream.allMatch(ModelStore::isModel);
                }

                if (isMultiModelsDirectory) {
                    // Check folders to see if they can be models as well
                    try (Stream<Path> stream = Files.list(modelStore)) {
                        urls.addAll(
                                stream.map(ModelStore::mapModelUrl)
                                        .filter(Objects::nonNull)
                                        .collect(Collectors.toList()));
                    }
                } else {
                    // Check if root model store folder contains a model
                    String url = mapModelUrl(modelStore);
                    if (url != null) {
                        urls.add(url);
                    }
                }
            } else {
                logger.warn("Model store path is not found: {}", modelStore);
            }
        } else {
            String[] modelsUrls = loadModels.split("[, ]+");
            urls.addAll(Arrays.asList(modelsUrls));
        }

        String huggingFaceModelId = Utils.getEnvOrSystemProperty("HF_MODEL_ID");
        if (huggingFaceModelId != null) {
            urls.add(createHuggingFaceModel(huggingFaceModelId));
        }

        for (String url : urls) {
            logger.info("Initializing model: {}", url);
            Matcher matcher = MODEL_STORE_PATTERN.matcher(url);
            if (!matcher.matches()) {
                throw new AssertionError("Invalid model store url: " + url);
            }
            String endpoint = matcher.group(2);
            String modelUrl = matcher.group(3);
            String version = null;
            String engineName = null;
            String deviceMapping = null;
            String modelName = null;
            if (endpoint != null) {
                String[] tokens = endpoint.split(":", -1);
                modelName = tokens[0];
                if (tokens.length > 1) {
                    version = tokens[1].isEmpty() ? null : tokens[1];
                }
                if (tokens.length > 2) {
                    engineName = tokens[2].isEmpty() ? null : tokens[2];
                }
                if (tokens.length > 3) {
                    deviceMapping = tokens[3];
                }
            }

            URI uri = WorkflowDefinition.toWorkflowUri(modelUrl);
            if (uri != null) {
                workflows.add(WorkflowDefinition.parse(modelName, uri).toWorkflow());
            } else {
                if (modelName == null) {
                    modelName = ModelInfo.inferModelNameFromUrl(modelUrl);
                }
                ModelInfo<Input, Output> modelInfo =
                        new ModelInfo<>(
                                modelName,
                                modelUrl,
                                version,
                                engineName,
                                deviceMapping,
                                Input.class,
                                Output.class,
                                -1,
                                -1,
                                -1,
                                -1,
                                -1,
                                -1);
                workflows.add(new Workflow(modelInfo));
            }
            startupModels.add(modelName);
        }
    }

    /**
     * Returns a list of workflows to be loaded on startup.
     *
     * @return a list of workflows to be loaded on startup
     */
    public List<Workflow> getWorkflows() {
        return workflows;
    }

    /**
     * Maps model directory to model url.
     *
     * @param path the model directory
     * @return the mapped model url
     */
    public static String mapModelUrl(Path path) {
        try {
            if (!isModel(path)) {
                return null;
            }
            try (Stream<Path> stream = Files.list(path)) {
                if (stream.findFirst().isEmpty()) {
                    return null;
                }
            }

            path = Utils.getNestedModelDir(path);
            String url = path.toUri().toURL().toString();
            String modelName = ModelInfo.inferModelNameFromUrl(url);
            logger.info("Found model {}={}", modelName, url);
            return modelName + '=' + url;
        } catch (MalformedURLException e) {
            throw new AssertionError("Invalid path: " + path, e);
        } catch (IOException e) {
            logger.warn("Failed to access file: {}", path, e);
            return null;
        }
    }

    private static boolean isModel(Path path) {
        String fileName = Objects.requireNonNull(path.getFileName()).toString();
        if (fileName.startsWith(".")) {
            return false;
        }
        return Files.exists(path)
                && (Files.isDirectory(path) || FilenameUtils.isArchiveFile(fileName));
    }

    private String createHuggingFaceModel(String modelId) throws IOException {
        if (modelId.startsWith("djl://") || modelId.startsWith("s3://")) {
            return modelId;
        }
        Path path = Paths.get(modelId);
        if (Files.exists(path)) {
            // modelId point to a local file
            return mapModelUrl(path);
        }

        // TODO: Download the full model from HF
        String hash = Utils.hash(modelId);
        String downloadDir = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = downloadDir == null ? Utils.getCacheDir() : Paths.get(downloadDir);
        Path huggingFaceModelDir = parent.resolve(hash);
        String modelName = modelId.replaceAll("(\\W|^_)", "_");
        if (Files.exists(huggingFaceModelDir)) {
            logger.warn("HuggingFace Model {} already exists, use random model name", modelId);
            return modelName + '_' + RandomUtils.nextInt() + '=' + huggingFaceModelDir;
        }
        String huggingFaceModelRevision = Utils.getEnvOrSystemProperty("HF_REVISION");
        Properties huggingFaceProperties = new Properties();
        huggingFaceProperties.put("option.model_id", modelId);
        if (huggingFaceModelRevision != null) {
            huggingFaceProperties.put("option.revision", huggingFaceModelRevision);
        }
        String task = Utils.getEnvOrSystemProperty("HF_TASK");
        if (task != null) {
            huggingFaceProperties.put("option.task", task);
        }
        Files.createDirectories(huggingFaceModelDir);
        Path propertiesFile = huggingFaceModelDir.resolve("serving.properties");
        try (BufferedWriter writer = Files.newBufferedWriter(propertiesFile)) {
            huggingFaceProperties.store(writer, null);
        }
        logger.debug("Created serving.properties for model at path {}", propertiesFile);
        return modelName + '=' + huggingFaceModelDir;
    }
}
