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
package ai.djl.serving.wlm;

import ai.djl.ModelException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;

import com.google.gson.JsonSyntaxException;
import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Properties;

/** A utility class to detect optimal engine for LMI model. */
public final class LmiUtils {

    private static final Logger logger = LoggerFactory.getLogger(LmiUtils.class);

    private static final List<String> DEEPSPEED_MODELS =
            List.of(
                    "bert",
                    "bloom",
                    "gpt_neo",
                    "gpt_neox",
                    "gpt2",
                    "gptj",
                    "opt",
                    "roberta",
                    "xlm-roberta");

    private LmiUtils() {}

    static String inferLmiEngine(ModelInfo<?, ?> modelInfo) throws ModelException {
        // MMS/Torchserve
        if (Files.isDirectory(modelInfo.modelDir.resolve("MAR-INF"))) {
            logger.info("Found legacy torchserve model, use Python engine.");
            return "Python";
        }

        Properties prop = modelInfo.prop;
        HuggingFaceModelConfig modelConfig = getHuggingFaceModelConfig(modelInfo);
        if (modelConfig == null) {
            logger.info("No config.json found, use Python engine.");
            return "Python";
        }

        String modelType = modelConfig.getModelType();
        String engineName;
        if ("stable-diffusion".equals(modelType)) {
            // TODO: Move this from hardcoded to deduced in PyModel
            prop.setProperty("option.entryPoint", "djl_python.stable-diffusion");
            engineName = "DeepSpeed";
        } else if (isDeepSpeedRecommended(modelType)) {
            engineName = "DeepSpeed";
        } else {
            engineName = "Python";
        }
        logger.info("Detected engine: {}, modelType: {}", engineName, modelType);
        return engineName;
    }

    private static URI generateHuggingFaceConfigUri(ModelInfo<?, ?> modelInfo, String modelId)
            throws ModelException, IOException {
        URI configUri = null;
        Path modelDir = modelInfo.modelDir;
        if (modelId != null && modelId.startsWith("s3://")) {
            // This is definitely suboptimal, but for the majority of cases we need to download this
            // s3 model eventually, so it is not the worst thing to download it now.
            modelInfo.downloadS3();
            Path downloadDir = modelInfo.downloadS3Dir;
            if (Files.isRegularFile(downloadDir.resolve("config.json"))) {
                configUri = downloadDir.resolve("config.json").toUri();
            } else if (Files.isRegularFile(downloadDir.resolve("model_index.json"))) {
                configUri = downloadDir.resolve("model_index.json").toUri();
            }
        } else if (modelId != null) {
            modelInfo.prop.setProperty("option.model_id", modelId);
            configUri = URI.create("https://huggingface.co/" + modelId + "/raw/main/config.json");
            HttpURLConnection configUrl = (HttpURLConnection) configUri.toURL().openConnection();
            // stable diffusion models have a different file name with the config... sometimes
            if (HttpURLConnection.HTTP_OK != configUrl.getResponseCode()) {
                configUri =
                        URI.create(
                                "https://huggingface.co/" + modelId + "/raw/main/model_index.json");
            }
        } else if (Files.isRegularFile(modelDir.resolve("config.json"))) {
            configUri = modelDir.resolve("config.json").toUri();
        } else if (Files.isRegularFile(modelDir.resolve("model_index.json"))) {
            configUri = modelDir.resolve("model_index.json").toUri();
        }
        return configUri;
    }

    private static HuggingFaceModelConfig getHuggingFaceModelConfig(ModelInfo<?, ?> modelInfo)
            throws ModelException {
        String modelId = modelInfo.prop.getProperty("option.model_id");
        if (modelId == null) {
            // Deprecated but for backwards compatibility
            modelId = modelInfo.prop.getProperty("option.s3url");
        }
        try {
            URI modelConfigUri = generateHuggingFaceConfigUri(modelInfo, modelId);
            if (modelConfigUri == null) {
                return null;
            }
            try (InputStream is = modelConfigUri.toURL().openStream()) {
                return JsonUtils.GSON.fromJson(Utils.toString(is), HuggingFaceModelConfig.class);
            }
        } catch (IOException | JsonSyntaxException e) {
            throw new ModelNotFoundException("Invalid huggingface model id: " + modelId, e);
        }
    }

    private static boolean isDeepSpeedRecommended(String modelType) {
        return isPythonDependencyInstalled("/usr/local/bin/deepspeed", "deepspeed")
                && DEEPSPEED_MODELS.contains(modelType);
    }

    private static boolean isPythonDependencyInstalled(
            String dependencyPath, String dependencyName) {
        Path expectedPath = Paths.get(dependencyPath);
        if (Files.exists(expectedPath)) {
            return true;
        }
        String[] cmd = {"pip", "list", "|", "grep", dependencyName};
        try {
            Process exec = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            String logOutput;
            try (InputStream is = exec.getInputStream()) {
                logOutput = Utils.toString(is);
            }
            int exitCode = exec.waitFor();
            if (exitCode == 0 && logOutput.contains(dependencyName)) {
                return true;
            } else {
                logger.warn("Did not find {} installed in python environment", dependencyName);
            }
        } catch (IOException | InterruptedException e) {
            logger.warn("pip check for {} failed", dependencyName, e);
        }
        return false;
    }

    // This represents  the config of huggingface models NLP models as well
    // as the config of diffusers models. The config is different for both, but for
    // now we can leverage a single class since we don't need too much information from the config.
    static final class HuggingFaceModelConfig {

        @SerializedName("model_type")
        private String modelType;

        @SerializedName("_diffusers_version")
        private String diffusersVersion;

        public String getModelType() {
            if (modelType == null) {
                return diffusersVersion == null ? null : "stable-diffusion";
            }
            return modelType;
        }
    }
}
