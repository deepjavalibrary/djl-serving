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

import java.util.Properties;
import java.util.Set;

/** A utility class to auto configure LMI model properties. */
public final class LmiConfigRecommender {

    private static final Logger logger = LoggerFactory.getLogger(LmiConfigRecommender.class);

    private static final Set<String> OPTIMIZED_TASK_ARCHITECTURES =
            Set.of("ForCausalLM", "LMHeadModel", "ForConditionalGeneration");

    private LmiConfigRecommender() {}

    static void configure(Properties lmiProperties, LmiUtils.HuggingFaceModelConfig modelConfig) {
        String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
        setRollingBatch(lmiProperties, modelConfig, features);
        setMpiMode(lmiProperties);
        setPipelineParallelDegree(lmiProperties);
        setTensorParallelDegree(lmiProperties);
        setRollingBatchSize(lmiProperties);
        setIsPeftModel(lmiProperties, modelConfig);
        setPropertiesForLora(lmiProperties);
        setPropertiesForStatefulSessions(lmiProperties);
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
        if ("lmi-dist".equals(rollingBatch)) {
            logger.warn(
                    "lmi-dist has been removed in DJL Serving 0.33.0. We recommend using vllm which"
                            + " serves as a drop-in replacement for most use-cases. Overriding"
                            + " specified rolling batch of lmi-dist to vllm");
            rollingBatch = "vllm";
            lmiProperties.setProperty("option.mpi_mode", "false");
        } else if (!"auto".equals(rollingBatch)) {
            return;
        } else if (!isTextGenerationModel(modelConfig)) {
            // Non text-generation use-cases are not compatible with rolling batch
            rollingBatch = "disable";
        } else if (isVllmEnabled(features)) {
            rollingBatch = "disable";
            lmiProperties.setProperty("option.async_mode", "true");
        } else if (isTrtLlmEnabled(features)) {
            rollingBatch = "disable";
            lmiProperties.setProperty("option.async_mode", "true");
        } else {
            rollingBatch = "disable";
        }
        lmiProperties.setProperty("option.rolling_batch", rollingBatch);
    }

    private static void setMpiMode(Properties lmiProperties) {
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch");
        if ("trtllm".equals(rollingBatch)) {
            lmiProperties.setProperty("option.mpi_mode", "true");
        }
    }

    private static void setTensorParallelDegree(Properties lmiProperties) {
        if (lmiProperties.containsKey("option.tensor_parallel_degree")) {
            return;
        }
        String tpDegree = Utils.getenv("TENSOR_PARALLEL_DEGREE", "max");
        int ppDegree =
                Integer.parseInt(lmiProperties.getProperty("option.pipeline_parallel_degree"));
        if ("max".equals(tpDegree)) {
            int numGpus = CudaUtils.getGpuCount();
            if (numGpus > 0) {
                tpDegree = String.valueOf(numGpus / ppDegree);

            } else {
                tpDegree = null;
            }
        }
        if (tpDegree != null) {
            lmiProperties.setProperty("option.tensor_parallel_degree", tpDegree);
        }
    }

    private static void setPipelineParallelDegree(Properties lmiProperties) {
        if (lmiProperties.containsKey("option.pipeline_parallel_degree")) {
            return;
        }
        String ppDegree = Utils.getenv("PIPELINE_PARALLEL_DEGREE", "1");
        lmiProperties.setProperty("option.pipeline_parallel_degree", ppDegree);
    }

    private static void setRollingBatchSize(Properties lmiProperties) {
        if (lmiProperties.containsKey("option.max_rolling_batch_size")) {
            return;
        }
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch");
        int rollingBatchSize = 32;
        if ("vllm".equals(rollingBatch)) {
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

    private static void setIsPeftModel(
            Properties lmiProperties, LmiUtils.HuggingFaceModelConfig modelConfig) {
        if (modelConfig.isPeftModel()) {
            lmiProperties.setProperty("option.is_peft_model", "true");
        }
    }

    private static void setPropertiesForLora(Properties lmiProperties) {
        // If option.enable_lora=true, set load_on_devices=0 and maxWorkers=1 because we
        // only
        // support one worker thread
        // for LoRA.
        // TODO: Support multiple worker threads for LoRA.
        boolean enableLora = Boolean.parseBoolean(lmiProperties.getProperty("option.enable_lora"));
        if (enableLora) {
            logger.info(
                    "option.enable_lora is set to true, setting load_on_devices=0 and"
                            + " maxWorkers=1");
            lmiProperties.setProperty("load_on_devices", "0");
            lmiProperties.setProperty("maxWorkers", "1");
        }
    }

    private static void setPropertiesForStatefulSessions(Properties lmiProperties) {
        // If option.enable_stateful_sessions=true, set load_on_devices=0 and
        // maxWorkers=1 because
        // we
        // only support one worker thread for stateful sessions.
        boolean enableStatefulSessions =
                Boolean.parseBoolean(lmiProperties.getProperty("option.enable_stateful_sessions"));
        if (enableStatefulSessions) {
            logger.info(
                    "option.enable_stateful_sessions is set to true,"
                            + " setting load_on_devices=0 and maxWorkers=1");
            lmiProperties.setProperty("load_on_devices", "0");
            lmiProperties.setProperty("maxWorkers", "1");
        }
    }

    private static boolean isVllmEnabled(String features) {
        return features != null && features.contains("vllm");
    }

    private static boolean isTrtLlmEnabled(String features) {
        return features != null && features.contains("trtllm");
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
