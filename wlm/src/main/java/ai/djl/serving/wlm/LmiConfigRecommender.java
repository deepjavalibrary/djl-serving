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

import java.util.Map;
import java.util.Properties;

/** A utility class to auto configure LMI model properties. */
public final class LmiConfigRecommender {

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

    private LmiConfigRecommender() {}

    static void configure(
            ModelInfo<?, ?> modelInfo,
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig) {
        String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
        setRollingBatch(lmiProperties, modelConfig, features);
        setEngine(lmiProperties, modelConfig, features);
        setTensorParallelDegree(lmiProperties);
        setDynamicBatch(modelInfo, lmiProperties, modelConfig, features);
    }

    private static void setRollingBatch(
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            String features) {
        String rollingBatch = lmiProperties.getProperty("option.rolling_batch", "auto");
        if (!"auto".equals(rollingBatch)) {
            return;
        } else if (isVLLMEnabled(features) && isLmiDistEnabled(features)) {
            rollingBatch = MODEL_TO_ROLLING_BATCH.getOrDefault(modelConfig.getModelType(), "auto");
        } else if (LmiUtils.isTrtLLM(lmiProperties)) {
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
        // TODO: Change it once TrtLLM supports T5 with inflight batching.
        if (isTrtLLMEnabled(features)
                && "disable".equals(rollingBatch)
                && "t5".equals(modelConfig.getModelType())) {
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
            ModelInfo<?, ?> modelInfo,
            Properties lmiProperties,
            LmiUtils.HuggingFaceModelConfig modelConfig,
            String features) {
        if (lmiProperties.containsKey("batch_size")
                || lmiProperties.containsKey("max_dynamic_batch_size")) {
            return;
        }
        // TODO: Change it once TrtLLM supports T5 with inflight batching.
        if ("t5".equals(modelConfig.getModelType()) && isTrtLLMEnabled(features)) {
            modelInfo.batchSize = 32;
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
}
