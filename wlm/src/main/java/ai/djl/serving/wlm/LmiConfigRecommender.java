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
    private static final Map<String, String> MODEL_TO_ROLLING_BATCH =
            Map.ofEntries(
                    Map.entry("falcon", "lmi-dist"),
                    Map.entry("gpt-neox", "lmi-dist"),
                    Map.entry("t5", "lmi-dist"),
                    Map.entry("llama", "lmi-dist"),
                    Map.entry("mpt", "lmi-dist"),
                    Map.entry("gpt-bigcode", "lmi-dist"),
                    Map.entry("aquila", "vllm"),
                    Map.entry("baichuan", "vllm"),
                    Map.entry("bloom", "vllm"),
                    Map.entry("chatglm", "vllm"),
                    Map.entry("deci", "vllm"),
                    Map.entry("gemma", "vllm"),
                    Map.entry("gpt2", "vllm"),
                    Map.entry("gptj", "vllm"),
                    Map.entry("internlm2", "vllm"),
                    Map.entry("mistral", "vllm"),
                    Map.entry("mixtral", "vllm"),
                    Map.entry("opt", "vllm"),
                    Map.entry("phi", "vllm"),
                    Map.entry("qwen", "vllm"),
                    Map.entry("qwen2", "vllm"),
                    Map.entry("stablelm", "vllm"));

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
        setEngine(lmiProperties, modelConfig, features);
        setTensorParallelDegree(lmiProperties);
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
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch", "auto");
        if (!"auto".equals(rollingBatch)) {
            return;
        } else if (!isTextGenerationModel(modelConfig)) {
            // Non text-generation use-cases are not compatible with rolling batch
            rollingBatch = "disable";
        } else if (isVLLMEnabled(features) && isLmiDistEnabled(features)) {
            rollingBatch = MODEL_TO_ROLLING_BATCH.getOrDefault(modelConfig.getModelType(), "auto");
        } else if (LmiUtils.isTrtLLMRollingBatch(lmiProperties)) {
            rollingBatch = "trtllm";
        }
        lmiProperties.setProperty("option.rolling_batch", rollingBatch);
    }

    private static void setEngine(
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            String features) {
        if (lmiProperties.containsKey("engine")) {
            return;
        }
        String engine = "Python";
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch");
        if ("lmi-dist".equals(rollingBatch) || "trtllm".equals(rollingBatch)) {
            engine = "MPI";
            lmiProperties.setProperty("option.mpi_mode", "true");
        }
        //  TODO TrtLLM python backend: Change it once TrtLLM supports T5 with inflight batching.
        if (isT5TrtLLM(modelConfig, features)) {
            engine = "MPI";
            lmiProperties.setProperty("option.mpi_mode", "true");
        }
        lmiProperties.setProperty("engine", engine);
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
        if (isT5TrtLLM(modelConfig, features)) {

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

    private static boolean isVLLMEnabled(String features) {
        return features != null && features.contains("vllm");
    }

    private static boolean isLmiDistEnabled(String features) {
        return features != null && features.contains("lmi-dist");
    }

    private static boolean isTrtLLMEnabled(String features) {
        return features != null && features.contains("trtllm");
    }

    private static boolean isT5TrtLLM(
            LmiUtils.HuggingFaceModelConfig modelConfig, String features) {
        return isTrtLLMEnabled(features) && "t5".equals(modelConfig.getModelType());
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
