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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import com.google.gson.JsonSyntaxException;
import com.google.gson.annotations.SerializedName;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Properties;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

/** A utility class to detect optimal engine for LMI model. */
public final class LmiUtils {

    private static final Logger logger = LoggerFactory.getLogger(LmiUtils.class);

    private LmiUtils() {}

    static void configureLmiModel(ModelInfo<?, ?> modelInfo) throws ModelException {
        HuggingFaceModelConfig modelConfig = getHuggingFaceModelConfig(modelInfo);
        Properties prop = modelInfo.getProperties();
        if (modelConfig == null) {
            // Precompiled models may not have config, set mpi mode when trtllm
            // TODO: Include TRT/Neuron compiled models in this configure flow
            String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
            if (features != null && features.contains("trtllm")) {
                prop.setProperty("option.mpi_mode", "true");
            }
            return;
        }

        LmiConfigRecommender.configure(modelInfo, prop, modelConfig);
        logger.info(
                "Detected mpi_mode: {}, rolling_batch: {}, tensor_parallel_degree {}, for"
                        + " modelType: {}",
                prop.getProperty("option.mpi_mode"),
                prop.getProperty("option.rolling_batch"),
                prop.getProperty("option.tensor_parallel_degree"),
                modelConfig.getModelType());
    }

    static boolean isTrtLlmRollingBatch(Properties properties) {
        String rollingBatch = properties.getProperty("option.rolling_batch");
        if ("trtllm".equals(rollingBatch)) {
            return true;
        }
        if (rollingBatch == null || "auto".equals(rollingBatch)) {
            // FIXME: find a better way to set default rolling batch for trtllm
            String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
            return features != null && features.contains("trtllm");
        }
        return false;
    }

    static boolean needConvert(ModelInfo<?, ?> info) {
        Properties properties = info.getProperties();
        return isTrtLlmRollingBatch(properties) || properties.containsKey("trtllm_python_backend");
    }

    static void convertTrtLLM(ModelInfo<?, ?> info) throws IOException {
        Path trtRepo;
        String modelId = null;
        if (info.downloadDir != null) {
            trtRepo = info.downloadDir;
        } else {
            trtRepo = info.modelDir;
            modelId = info.prop.getProperty("option.model_id");
            if (modelId != null && Files.isDirectory(Paths.get(modelId))) {
                trtRepo = Paths.get(modelId);
            }
        }

        if (modelId == null) {
            modelId = trtRepo.toString();
        }

        String tpDegree = info.prop.getProperty("option.tensor_parallel_degree");
        if (tpDegree == null) {
            tpDegree = Utils.getenv("TENSOR_PARALLEL_DEGREE", "max");
        }
        if ("max".equals(tpDegree)) {
            tpDegree = String.valueOf(CudaUtils.getGpuCount());
        }

        String ppDegree = info.prop.getProperty("option.pipeline_parallel_degree");
        if (ppDegree == null) {
            ppDegree = Utils.getenv("PIPELINE_PARALLEL_DEGREE", "1");
        }

        // TODO TrtLLM python backend: Change it once TrtLLM supports T5 with inflight batching.
        if (info.prop.containsKey("trtllm_python_backend")) {
            // Inflight batching support is not available for certain models like t5.
            // Python backend models have different model repo format compared to C++ backend.
            // And whether it is valid or not is checked in tensorrt_llm_toolkit. So it is not
            // necessary to check here.
            if (!isValidTrtLlmPythonModelRepo(trtRepo)) {
                info.downloadDir = buildTrtLlmArtifacts(info.modelDir, modelId, tpDegree, ppDegree);
            }
        } else {
            info.prop.put("option.rolling_batch", "trtllm");
            if (!isValidTrtLlmModelRepo(trtRepo)) {
                info.downloadDir = buildTrtLlmArtifacts(info.modelDir, modelId, tpDegree, ppDegree);
            }
        }
    }

    static void convertOnnxModel(ModelInfo<?, ?> info) throws IOException {
        String prefix = info.prop.getProperty("option.modelName", info.modelDir.toFile().getName());
        if (Files.isRegularFile(info.modelDir)
                || Files.isRegularFile(info.modelDir.resolve(prefix + ".onnx"))
                || Files.isRegularFile(info.modelDir.resolve("model.onnx"))) {
            return;
        }

        Path repo;
        String modelId = null;
        if (info.downloadDir != null) {
            repo = info.downloadDir;
        } else {
            repo = info.modelDir;
            modelId = info.prop.getProperty("option.model_id");
            if (modelId != null && Files.isDirectory(Paths.get(modelId))) {
                repo = Paths.get(modelId);
            }
        }

        if (modelId == null) {
            modelId = repo.toString();
        }
        String optimization = info.prop.getProperty("option.optimization");
        info.modelUrl = convertOnnx(modelId, optimization).toUri().toURL().toString();
    }

    private static Path convertOnnx(String modelId, String optimization) throws IOException {
        logger.info("Converting model to onnx artifacts");
        String hash = Utils.hash(modelId);
        String download = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = download == null ? Utils.getCacheDir() : Paths.get(download);
        Path repoDir = parent.resolve("onnx").resolve(hash);
        if (Files.exists(repoDir)) {
            logger.info("Onnx artifacts already converted: {}", repoDir);
            return repoDir;
        }

        Engine onnx = Engine.getEngine("OnnxRuntime");
        boolean hasCuda = onnx.getGpuCount() > 0;
        if (optimization == null || optimization.isBlank()) {
            optimization = hasCuda ? "O4" : "O2";
        } else if (!optimization.matches("O\\d")) {
            throw new IllegalArgumentException("Unsupported optimization level: " + optimization);
        }

        String[] cmd = {
            "djl-convert",
            "--output-dir",
            repoDir.toAbsolutePath().toString(),
            "--output-format",
            "OnnxRuntime",
            "-m",
            modelId,
            "--optimize",
            optimization,
            "--device",
            hasCuda ? "cuda" : "cpu"
        };
        boolean success = false;
        try {
            Process exec = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(exec.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logger.debug("convert: {}", line);
                }
            }
            int exitCode = exec.waitFor();
            if (0 != exitCode) {
                throw new EngineException("Model conversion process failed!");
            }
            success = true;
            logger.info("Onnx artifacts built successfully");
            return repoDir;
        } catch (InterruptedException e) {
            throw new IOException("Failed to build Onnx artifacts", e);
        } finally {
            if (!success) {
                Utils.deleteQuietly(repoDir);
            }
        }
    }

    /**
     * Returns the Huggingface config.json file URI.
     *
     * @param modelInfo the model object
     * @param modelId the model id
     * @return the Huggingface config.json file URI
     */
    public static URI generateHuggingFaceConfigUri(ModelInfo<?, ?> modelInfo, String modelId) {
        Path modelDir = modelInfo.modelDir;
        if (Files.isRegularFile(modelDir.resolve("config.json"))) {
            return modelDir.resolve("config.json").toUri();
        } else if (Files.isRegularFile(modelDir.resolve("model_index.json"))) {
            return modelDir.resolve("model_index.json").toUri();
        }
        if (modelId == null || modelId.startsWith("djl://")) {
            // djl model should be covered by local file in modelDir case
            return null;
        } else if (modelId.startsWith("s3://")) {
            // HF_MODEL_ID=s3:// should not reach here, this is OPTION_MODEL_ID case.
            Path downloadDir = modelInfo.downloadDir;
            if (Files.isRegularFile(downloadDir.resolve("config.json"))) {
                return downloadDir.resolve("config.json").toUri();
            } else if (Files.isRegularFile(downloadDir.resolve("model_index.json"))) {
                return downloadDir.resolve("model_index.json").toUri();
            }
            return null;
        } else {
            modelInfo.prop.setProperty("option.model_id", modelId);
            Path dir = Paths.get(modelId);
            if (Files.isDirectory(dir)) {
                Path configFile = dir.resolve("config.json");
                if (Files.isRegularFile(configFile)) {
                    return configFile.toUri();
                }
                // stable diffusion models have a different file name with the config...
                configFile = dir.resolve("model_index.json");
                if (Files.isRegularFile(configFile)) {
                    return configFile.toUri();
                }
                return null;
            }
            return getHuggingFaceHubConfigUri(modelId);
        }
    }

    private static URI getHuggingFaceHubConfigUri(String modelId) {
        String[] possibleConfigFiles = {"config.json", "model_index.json"};
        String hubToken = Utils.getEnvOrSystemProperty("HF_TOKEN");
        Map<String, String> headers = new ConcurrentHashMap<>();
        headers.put("User-Agent", "DJL/" + Engine.getDjlVersion());
        if (hubToken != null) {
            headers.put("Authorization", "Bearer " + hubToken);
        }
        for (String configFile : possibleConfigFiles) {
            URI configUri =
                    URI.create("https://huggingface.co/" + modelId + "/raw/main/" + configFile);
            try (InputStream is = Utils.openUrl(configUri.toURL(), headers)) {
                is.transferTo(OutputStream.nullOutputStream());
                return configUri;
            } catch (IOException e) {
                logger.warn("Hub config file {} does not exist for model {}.", configFile, modelId);
            }
        }
        return null;
    }

    private static HuggingFaceModelConfig getHuggingFaceModelConfig(ModelInfo<?, ?> modelInfo)
            throws ModelException {
        String modelId = modelInfo.prop.getProperty("option.model_id");
        URI modelConfigUri = generateHuggingFaceConfigUri(modelInfo, modelId);
        if (modelConfigUri == null) {
            return null;
        }
        Map<String, String> headers = new ConcurrentHashMap<>();
        headers.put("User-Agent", "DJL/" + Engine.getDjlVersion());
        String hubToken = Utils.getEnvOrSystemProperty("HF_TOKEN");
        if (hubToken != null) {
            headers.put("Authorization", "Bearer " + hubToken);
        }

        try (InputStream is = Utils.openUrl(modelConfigUri.toURL(), headers)) {
            return JsonUtils.GSON.fromJson(Utils.toString(is), HuggingFaceModelConfig.class);
        } catch (IOException | JsonSyntaxException e) {
            throw new ModelNotFoundException("Invalid huggingface model id: " + modelId, e);
        }
    }

    private static Path buildTrtLlmArtifacts(
            Path modelDir, String modelId, String tpDegree, String ppDegree) throws IOException {
        logger.info("Converting model to TensorRT-LLM artifacts");
        String hash = Utils.hash(modelId + tpDegree);
        String download = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = download == null ? Utils.getCacheDir() : Paths.get(download);
        Path trtLlmRepoDir = parent.resolve("trtllm").resolve(hash);
        if (Files.exists(trtLlmRepoDir)) {
            logger.info("TensorRT-LLM artifacts already converted: {}", trtLlmRepoDir);
            return trtLlmRepoDir;
        }

        String[] cmd = {
            "python",
            "/opt/djl/partition/trt_llm_partition.py",
            "--properties_dir",
            modelDir.toAbsolutePath().toString(),
            "--trt_llm_model_repo",
            trtLlmRepoDir.toString(),
            "--tensor_parallel_degree",
            tpDegree,
            "--pipeline_parallel_degree",
            ppDegree,
            "--model_path",
            modelId
        };
        boolean success = false;
        try {
            Process exec = new ProcessBuilder(cmd).redirectErrorStream(true).start();
            try (BufferedReader reader =
                    new BufferedReader(
                            new InputStreamReader(exec.getInputStream(), StandardCharsets.UTF_8))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    logger.info("convert_py: {}", line);
                }
            }
            int exitCode = exec.waitFor();
            if (0 != exitCode) {
                throw new EngineException("Model conversion process failed!");
            }
            success = true;
            logger.info("TensorRT-LLM artifacts built successfully");
            return trtLlmRepoDir;
        } catch (InterruptedException e) {
            throw new IOException("Failed to build TensorRT-LLM artifacts", e);
        } finally {
            if (!success) {
                Utils.deleteQuietly(trtLlmRepoDir);
            }
        }
    }

    // TODO: migrate this to CUDAUtils in next version
    static String getAWSGpuMachineType() {
        String computeCapability = CudaUtils.getComputeCapability(0);
        // Get gpu memory in GB sizes
        double totalMemory =
                CudaUtils.getGpuMemory(Device.gpu()).getMax() / 1024.0 / 1024.0 / 1024.0;
        if ("7.5".equals(computeCapability)) {
            return "g4";
        } else if ("8.0".equals(computeCapability)) {
            if (totalMemory > 45.0) {
                return "p4de";
            }
            return "p4d";
        } else if ("8.6".equals(computeCapability)) {
            return "g5";
        } else if ("8.9".equals(computeCapability)) {
            if (totalMemory > 25.0) {
                return "g6e";
            }
            return "g6";
        } else if ("9.0".equals(computeCapability)) {
            return "p5";
        } else {
            logger.warn("Could not identify GPU arch {}", computeCapability);
            return null;
        }
    }

    static boolean isValidTrtLlmPythonModelRepo(Path modelPath) throws IOException {
        AtomicBoolean isValid = new AtomicBoolean();
        try (Stream<Path> walk = Files.list(modelPath)) {
            walk.filter(Files::isDirectory)
                    .filter(
                            p -> {
                                String directoryName = p.getFileName().toString();
                                return directoryName.contains("encoder")
                                        || directoryName.contains("decoder");
                            })
                    .forEach(
                            p -> {
                                logger.info(String.valueOf(p));
                                Path configFile = p.resolve("config.json");
                                if (Files.isRegularFile(configFile)) {
                                    logger.info("Found trtllm python model: {}", p);
                                    isValid.set(true);
                                }
                            });
        }
        return isValid.get();
    }

    static boolean isValidTrtLlmModelRepo(Path modelPath) throws IOException {
        // TODO: match model name
        AtomicBoolean isValid = new AtomicBoolean();
        try (Stream<Path> walk = Files.list(modelPath)) {
            walk.filter(Files::isDirectory)
                    .forEach(
                            p -> {
                                Path confFile = p.resolve("config.pbtxt");
                                // TODO: add stricter check for tokenizer
                                Path tokenizer = p.resolve("tokenizer_config.json");
                                if (Files.isRegularFile(confFile)
                                        && Files.isRegularFile(tokenizer)) {
                                    logger.info("Found triton model: {}", p);
                                    isValid.set(true);
                                }
                            });
        }
        return isValid.get();
    }

    // This represents  the config of huggingface models NLP models as well
    // as the config of diffusers models. The config is different for both, but for
    // now we can leverage a single class since we don't need too much information from the config.
    static final class HuggingFaceModelConfig {

        @SerializedName("model_type")
        private String modelType;

        @SerializedName("architectures")
        private List<String> configArchitectures;

        @SerializedName("auto_map")
        private Map<String, String> autoMap;

        @SerializedName("_diffusers_version")
        private String diffusersVersion;

        private Set<String> allArchitectures;

        public String getModelType() {
            if (modelType == null) {
                return diffusersVersion == null ? null : "stable-diffusion";
            }
            return modelType;
        }

        public Set<String> getArchitectures() {
            if (allArchitectures == null) {
                determineAllArchitectures();
            }
            return allArchitectures;
        }

        private void determineAllArchitectures() {
            allArchitectures = new HashSet<>();
            if (configArchitectures != null) {
                allArchitectures.addAll(configArchitectures);
            }
            if (autoMap != null) {
                allArchitectures.addAll(autoMap.keySet());
            }
        }
    }
}
