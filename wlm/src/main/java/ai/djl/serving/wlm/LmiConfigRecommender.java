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

import ai.djl.util.Ec2Utils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Properties;
import java.util.Set;

/** A utility class to auto configure LMI model properties. */
public final class LmiConfigRecommender {

    private static final Logger logger = LoggerFactory.getLogger(LmiConfigRecommender.class);
    // TODO: model list is up to date with vLLM 0.4.2
    private static final Map<String, String> MODEL_TO_ROLLING_BATCH =
            Map.ofEntries(
                    Map.entry("falcon", "lmi-dist"),
                    Map.entry("gpt-neox", "lmi-dist"),
                    Map.entry("t5", "lmi-dist"),
                    Map.entry("llama", "lmi-dist"),
                    Map.entry("mpt", "lmi-dist"),
                    Map.entry("gpt-bigcode", "lmi-dist"),
                    Map.entry("aquila", "lmi-dist"),
                    Map.entry("baichuan", "lmi-dist"),
                    Map.entry("bloom", "lmi-dist"),
                    Map.entry("chatglm", "lmi-dist"),
                    Map.entry("cohere", "lmi-dist"),
                    Map.entry("dbrx", "lmi-dist"),
                    Map.entry("deci", "lmi-dist"),
                    Map.entry("gemma", "lmi-dist"),
                    Map.entry("gpt2", "lmi-dist"),
                    Map.entry("gptj", "lmi-dist"),
                    Map.entry("internlm", "lmi-dist"),
                    Map.entry("internlm2", "lmi-dist"),
                    Map.entry("jais", "lmi-dist"),
                    Map.entry("mistral", "lmi-dist"),
                    Map.entry("mixtral", "lmi-dist"),
                    Map.entry("opt", "lmi-dist"),
                    Map.entry("phi", "lmi-dist"),
                    Map.entry("phi3", "lmi-dist"),
                    Map.entry("qwen", "lmi-dist"),
                    Map.entry("qwen2", "lmi-dist"),
                    Map.entry("qwen2_moe", "lmi-dist"),
                    Map.entry("stablelm", "lmi-dist"),
                    Map.entry("xverse", "lmi-dist"),
                    Map.entry("starcoder2", "lmi-dist"));

    private static final Set<String> OPTIMIZED_TASK_ARCHITECTURES =
            Set.of("ForCausalLM", "LMHeadModel", "ForConditionalGeneration");

    private LmiConfigRecommender() {}

    static void configure(
            ModelInfo<?, ?> modelInfo,
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig) {
        String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
        setDynamicBatch(lmiProperties, modelConfig, modelInfo, features);
        setRollingBatch(lmiProperties, modelConfig, features);
        setMpiMode(lmiProperties, modelConfig, features);
        setTensorParallelDegree(lmiProperties);
        setRollingBatchSize(lmiProperties);
    }

    private static void setRollingBatch(
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            String features) {
        // If dynamic batch is enabled, we don't enable rolling batch.
        if (Integer.parseInt(lmiProperties.getProperty("batch_size", "1")) > 1) {
            lmiProperties.setProperty("option.rolling_batch", "disable");
            return;
        }

        String defaultRollingBatch = isTnxEnabled(features) ? "disable" : "auto";
        String rollingBatch =
                lmiProperties.getProperty("option.rolling_batch", defaultRollingBatch);
        String modelType = modelConfig.getModelType();
        if (!"auto".equals(rollingBatch)) {
            return;
        } else if (!isTextGenerationModel(modelConfig)) {
            // Non text-generation use-cases are not compatible with rolling batch
            rollingBatch = "disable";
        } else if (isTnxEnabled(features)) {
            rollingBatch = "tnx";
        } else if (isLmiDistEnabled(features)
                && "lmi-dist".equals(MODEL_TO_ROLLING_BATCH.get(modelType))) {
            rollingBatch = "lmi-dist";
        } else if (isVllmEnabled(features)
                && "vllm".equals(MODEL_TO_ROLLING_BATCH.get(modelType))) {
            rollingBatch = "vllm";
        } else if (isTrtLlmEnabled(features)) {
            rollingBatch = "trtllm";
        } else if (Ec2Utils.isSageMaker()) {
            rollingBatch = "scheduler";
        } else {
            rollingBatch = "disable";
        }
        lmiProperties.setProperty("option.rolling_batch", rollingBatch);
    }

    private static void setMpiMode(
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            String features) {
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch");
        if ("lmi-dist".equals(rollingBatch) || "trtllm".equals(rollingBatch)) {
            lmiProperties.setProperty("option.mpi_mode", "true");
        }
        //  TODO TrtLLM python backend: Change it once TrtLLM supports T5 with inflight batching.
        if (isT5TrtLlm(modelConfig, features)) {
            lmiProperties.setProperty("option.mpi_mode", "true");
        }
    }

    private static void setTensorParallelDegree(Properties lmiProperties) {
        if (lmiProperties.containsKey("option.tensor_parallel_degree")) {
            return;
        }
        String tpDegree = Utils.getenv("TENSOR_PARALLEL_DEGREE", "max");
        if ("max".equals(tpDegree)) {
            tpDegree = String.valueOf(CudaUtils.getGpuCount());
        }
        lmiProperties.setProperty("option.tensor_parallel_degree", tpDegree);
    }

    private static void setDynamicBatch(
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            ModelInfo<?, ?> modelInfo,
            String features) {
        // TODO TrtLLM python backend: Change it once TrtLLM supports T5 with inflight batching.
        if (isT5TrtLlm(modelConfig, features)) {
            // To do runtime compilation for TensorRT-LLM T5 model.
            lmiProperties.setProperty("trtllm_python_backend", String.valueOf(true));
            lmiProperties.setProperty("option.rolling_batch", "disable");

            // We set batch_size only when customer did not provide it.
            if (Integer.parseInt(lmiProperties.getProperty("batch_size", "0")) == 0) {
                modelInfo.batchSize = 32;
                lmiProperties.setProperty("batch_size", String.valueOf(32));
            }
        }
    }

    private static void setRollingBatchSize(Properties lmiProperties) {
        if (lmiProperties.containsKey("option.max_rolling_batch_size")) {
            return;
        }
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch");
        int rollingBatchSize = 32;
        if ("vllm".equals(rollingBatch) || "lmi-dist".equals(rollingBatch)) {
            rollingBatchSize = 256;
        }
        if ("trtllm".equals(rollingBatch)) {
            // https://github.com/NVIDIA/TensorRT-LLM/blob/v0.9.0/tensorrt_llm/_common.py#L208-L215
            // TODO: setting better default per 0.9.0 guidance 1024 * 16 = 16384
            if (!lmiProperties.containsKey("option.max_num_tokens")) {
                lmiProperties.setProperty("option.max_num_tokens", "16384");
            }
            rollingBatchSize = 256;
        }
        lmiProperties.setProperty(
                "option.max_rolling_batch_size", String.valueOf(rollingBatchSize));
    }

    private static boolean isVllmEnabled(String features) {
        return features != null && features.contains("vllm");
    }

    private static boolean isLmiDistEnabled(String features) {
        return features != null && features.contains("lmi-dist");
    }

    private static boolean isTrtLlmEnabled(String features) {
        return features != null && features.contains("trtllm");
    }

    private static boolean isTnxEnabled(String features) {
        return features != null && features.contains("tnx");
    }

    private static boolean isT5TrtLlm(
            LmiUtils.HuggingFaceModelConfig modelConfig, String features) {
        return isTrtLlmEnabled(features) && "t5".equals(modelConfig.getModelType());
    }

    private static boolean isTextGenerationModel(LmiUtils.HuggingFaceModelConfig modelConfig) {
        for (String arch : modelConfig.getArchitectures()) {
            boolean isTextGenerationModel =
                    OPTIMIZED_TASK_ARCHITECTURES.stream().anyMatch(arch::endsWith);
            if (isTextGenerationModel) {
                return true;
            }
        }
        logger.warn(
                "The model task architecture {} is not supported for optimized inference. LMI will"
                    + " attempt to load the model using HuggingFace Accelerate. Optimized inference"
                    + " performance is only available for the following task architectures: {}",
                modelConfig.getArchitectures(),
                OPTIMIZED_TASK_ARCHITECTURES);
        return false;
    }
}
