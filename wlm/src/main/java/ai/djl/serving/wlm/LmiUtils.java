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
import ai.djl.util.cuda.CudaUtils;

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
import java.util.Objects;
import java.util.Properties;

/** A utility class to detect optimal engine for LMI model. */
public final class LmiUtils {

    private static final Logger logger = LoggerFactory.getLogger(LmiUtils.class);

    private static final List<String> DEEPSPEED_MODELS =
            List.of(
                    "roberta",
                    "xlm-roberta",
                    "gpt2",
                    "bert",
                    "gpt_neo",
                    "gptj",
                    "opt",
                    "gpt_neox",
                    "bloom");

    private static final List<String> FASTERTRANSFORMER_MODELS = List.of("t5");

    private LmiUtils() {}

    static String inferLmiEngine(ModelInfo<?, ?> modelInfo) throws ModelException {
        // MMS/Torchserve
        if (Files.isDirectory(modelInfo.modelDir.resolve("MAR-INF"))) {
            return "Python";
        }

        Properties prop = modelInfo.prop;
        HuggingFaceModelConfig modelConfig = getHuggingFaceModelConfig(modelInfo);
        if (modelConfig == null) {
            return "Python";
        }

        int tensorParallelDegree;
        if (Utils.getEnvOrSystemProperty("TENSOR_PARALLEL_DEGREE") != null) {
            tensorParallelDegree =
                    Integer.parseInt(Utils.getEnvOrSystemProperty("TENSOR_PARALLEL_DEGREE"));
        } else if (prop.getProperty("option.tensor_parallel_degree") != null) {
            tensorParallelDegree =
                    Integer.parseInt(prop.getProperty("option.tensor_parallel_degree"));
        } else {
            // TODO: Assume use all GPUs for TP
            tensorParallelDegree = CudaUtils.getGpuCount();
        }

        if (tensorParallelDegree > 0) {
            prop.setProperty("option.tensor_parallel_degree", String.valueOf(tensorParallelDegree));
        }
        String huggingFaceTask = Utils.getEnvOrSystemProperty("HF_TASK");
        if (huggingFaceTask != null) {
            prop.setProperty("option.task", huggingFaceTask);
        }

        if ("stable-diffusion".equals(modelConfig.getModelType())) {
            // TODO: Move this from hardcoded to deduced in PyModel
            prop.setProperty("option.entryPoint", "djl_python.stable-diffusion");
            return "DeepSpeed";
        }

        if (!isTensorParallelSupported(modelConfig.getNumAttentionHeads(), tensorParallelDegree)) {
            return "Python";
        }
        if (isDeepSpeedRecommended(modelConfig.getModelType())) {
            return "DeepSpeed";
        }
        if (isFasterTransformerRecommended(modelConfig.getModelType())) {
            return "FasterTransformer";
        }
        return "Python";
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
        String modelId = Utils.getEnvOrSystemProperty("HF_MODEL_ID");
        if (modelId == null) {
            modelId = modelInfo.prop.getProperty("option.model_id");
        }
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

    private static boolean isFasterTransformerRecommended(String modelType) {
        return isPythonDependencyInstalled(
                        "/usr/local/backends/fastertransformer", "fastertransformer")
                && FASTERTRANSFORMER_MODELS.contains(modelType);
    }

    private static boolean isDeepSpeedRecommended(String modelType) {
        return isPythonDependencyInstalled("/usr/local/bin/deepspeed", "deepspeed")
                && DEEPSPEED_MODELS.contains(modelType);
    }

    private static boolean isTensorParallelSupported(
            long numAttentionHeads, int tensorParallelDegree) {
        return tensorParallelDegree > 0 && numAttentionHeads % tensorParallelDegree == 0;
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

        @SerializedName(
                value = "num_attention_heads",
                alternate = {"n_head", "num_heads"})
        private Long numAttentionHeads;

        public long getNumAttentionHeads() {
            return Objects.requireNonNullElse(numAttentionHeads, Long.MAX_VALUE);
        }

        public String getModelType() {
            if (modelType == null) {
                return diffusersVersion == null ? null : "stable-diffusion";
            }
            return modelType;
        }
    }
}
