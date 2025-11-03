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

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
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

    static void exec(List<String> cmd) throws IOException, InterruptedException {
        Process exec = new ProcessBuilder(cmd).redirectErrorStream(true).start();
        String logOutput;
        try (InputStream is = exec.getInputStream()) {
            logOutput = Utils.toString(is);
        }
        int exitCode = exec.waitFor();
        if (0 != exitCode || logOutput.startsWith("ERROR ")) {
            logger.error("{}", logOutput);
            throw new EngineException("Failed to execute: [" + String.join(" ", cmd) + "]");
        } else {
            logger.debug("{}", logOutput);
        }
    }

    static void configureLmiModel(ModelInfo<?, ?> modelInfo) throws ModelException {
        HuggingFaceModelConfig modelConfig = getHuggingFaceModelConfig(modelInfo);
        Properties prop = modelInfo.getProperties();
        if (modelConfig == null) {
            // Precompiled models may not have config, set mpi mode when trtllm
            // TODO: Include TRT compiled models in this configure flow
            String features = Utils.getEnvOrSystemProperty("SERVING_FEATURES");
            if (features != null && features.contains("trtllm")) {
                prop.setProperty("option.mpi_mode", "true");
            }
            logger.warn(
                    "Unable to fetch the HuggingFace Model Config for the specified model. If the"
                        + " model is a compiled model, or not a HuggingFace Pretrained Model, this"
                        + " is expected. If this is a HuggingFace Pretrained model, ensure that the"
                        + " model artifacts contain a config.json or params.json file. If you are"
                        + " using a HuggingFace Hub modelId, ensure that you are providing the"
                        + " HF_TOKEN environment variable for gated models.");
            return;
        }

        LmiConfigRecommender.configure(prop, modelConfig);
        logger.info(
                "Detected mpi_mode: {}, rolling_batch: {}, tensor_parallel_degree: {}, for"
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

    static boolean needConvertTrtLLM(ModelInfo<?, ?> info) {
        Properties properties = info.getProperties();
        return isTrtLlmRollingBatch(properties);
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

        info.prop.put("option.rolling_batch", "trtllm");
        if (!isValidTrtLlmModelRepo(trtRepo)) {
            info.downloadDir = buildTrtLlmArtifacts(info.prop, modelId, tpDegree, ppDegree);
        }
    }

    static boolean needConvertOnnx(ModelInfo<?, ?> info) {
        String prefix = info.prop.getProperty("option.modelName", info.modelDir.toFile().getName());
        // modelDir could be file:///model.onnx
        return !Files.isRegularFile(info.modelDir)
                && !prefix.endsWith(".onnx")
                && !Files.isRegularFile(info.modelDir.resolve(prefix + ".onnx"))
                && !Files.isRegularFile(info.modelDir.resolve("model.onnx"));
    }

    static void convertOnnxModel(ModelInfo<?, ?> info) throws IOException {
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
        boolean trustRemoteCode = "true".equals(info.prop.getProperty("option.trust_remote_code"));
        info.resolvedModelUrl =
                convertOnnx(modelId, optimization, trustRemoteCode).toUri().toURL().toString();
    }

    private static Path convertOnnx(String modelId, String optimization, boolean trustRemoteCode)
            throws IOException {
        logger.info("Converting model to onnx artifacts");
        String hash = Utils.hash(modelId);
        String download = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = download == null ? Utils.getCacheDir() : Paths.get(download);
        Path repoDir = parent.resolve("onnx").resolve(hash);
        if (Files.exists(repoDir)) {
            logger.info("Onnx artifacts already converted: {}", repoDir);
            return repoDir;
        }

        boolean hasCuda = CudaUtils.getGpuCount() > 0;
        if (optimization == null || optimization.isBlank()) {
            optimization = hasCuda ? "O4" : "O2";
        } else if (!optimization.matches("O\\d")) {
            throw new IllegalArgumentException("Unsupported optimization level: " + optimization);
        }

        List<String> cmd = new ArrayList<>();
        cmd.add("djl-convert");
        cmd.add("--output-dir");
        cmd.add(repoDir.toAbsolutePath().toString());
        cmd.add("--output-format");
        cmd.add("OnnxRuntime");
        cmd.add("-m");
        cmd.add(modelId);
        cmd.add("--optimize");
        cmd.add(optimization);
        cmd.add("--device");
        cmd.add(hasCuda ? "cuda" : "cpu");
        if (trustRemoteCode) {
            cmd.add("--trust-remote-code");
        }

        boolean success = false;
        try {
            logger.info("Converting model to onnx artifacts: {}", (Object) cmd);
            exec(cmd);
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

    static boolean needConvertRust(ModelInfo<?, ?> info) {
        return !Files.isRegularFile(info.modelDir.resolve("model.safetensors"))
                && (info.downloadDir == null
                        || !Files.isRegularFile(info.downloadDir.resolve("model.safetensors")));
    }

    static void convertRustModel(ModelInfo<?, ?> info) throws IOException {
        String modelId = info.prop.getProperty("option.model_id");
        boolean trustRemoteCode = "true".equals(info.prop.getProperty("option.trust_remote_code"));
        if (modelId == null) {
            logger.info("model_id not defined, skip rust model conversion.");
            return;
        }

        String hash = Utils.hash(modelId);
        String download = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = download == null ? Utils.getCacheDir() : Paths.get(download);
        Path repoDir = parent.resolve("rust").resolve(hash);
        if (Files.exists(repoDir)) {
            logger.info("Rust artifacts already converted: {}", repoDir);
            info.resolvedModelUrl = repoDir.toUri().toURL().toString();
            return;
        }

        List<String> cmd = new ArrayList<>();
        cmd.add("djl-convert");
        cmd.add("--output-dir");
        cmd.add(repoDir.toAbsolutePath().toString());
        cmd.add("--output-format");
        cmd.add("Rust");
        cmd.add("-m");
        cmd.add(modelId);
        if (trustRemoteCode) {
            cmd.add("--trust-remote-code");
        }

        boolean success = false;
        try {
            logger.info("Converting model to rust artifacts: {}", (Object) cmd);
            exec(cmd);
            success = true;
            logger.info("Rust artifacts built successfully");
            info.resolvedModelUrl = repoDir.toUri().toURL().toString();
        } catch (InterruptedException e) {
            throw new IOException("Failed to build Rust artifacts", e);
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
        String[] possibleConfigFiles = {
            "config.json", "adapter_config.json", "model_index.json", "params.json"
        };
        URI configUri;
        for (String configFile : possibleConfigFiles) {
            configUri = findHuggingFaceConfigUriForConfigFile(modelInfo, modelId, configFile);
            if (configUri != null) {
                return configUri;
            }
        }
        logger.debug("Did not find huggingface config file for model");
        return null;
    }

    private static URI findHuggingFaceConfigUriForConfigFile(
            ModelInfo<?, ?> modelInfo, String modelId, String configFile) {
        Path modelDir = modelInfo.modelDir;
        Path downloadDir = modelInfo.downloadDir;
        if (Files.isRegularFile(modelDir.resolve(configFile))) {
            logger.debug("Found config file: {} in modelDir {}", configFile, modelDir);
            return modelDir.resolve(configFile).toUri();
        }
        if (modelId == null || modelId.startsWith("djl://")) {
            // djl model should be covered by local file in modelDir case
            return null;
        }
        if (modelId.startsWith("s3://")) {
            // HF_MODEL_ID=s3:// should not reach here, this is OPTION_MODEL_ID case.
            if (Files.isRegularFile(downloadDir.resolve(configFile))) {
                logger.debug("Found config file: {} in downloadDir {}", configFile, downloadDir);
                return downloadDir.resolve(configFile).toUri();
            }
            return null;
        }
        Path maybeModelDir = Paths.get(modelId);
        if (Files.isDirectory(maybeModelDir)) {
            Path configFilePath = maybeModelDir.resolve(configFile);
            if (Files.isRegularFile(configFilePath)) {
                logger.debug(
                        "Found config file: {} in dir specified by modelId {}",
                        configFile,
                        maybeModelDir);
                return configFilePath.toUri();
            }
            return null;
        }
        return getHuggingFaceHubConfigUriFromHub(modelId, configFile);
    }

    private static URI getHuggingFaceHubConfigUriFromHub(String modelId, String configFile) {
        String hubToken = Utils.getEnvOrSystemProperty("HF_TOKEN");
        Map<String, String> headers = new ConcurrentHashMap<>();
        headers.put("User-Agent", "DJL/" + Engine.getDjlVersion());
        if (hubToken != null) {
            headers.put("Authorization", "Bearer " + hubToken);
        }
        URI configUri =
                URI.create("https://huggingface.co/" + modelId + "/resolve/main/" + configFile);
        try (InputStream is = Utils.openUrl(configUri.toURL(), headers)) {
            is.transferTo(OutputStream.nullOutputStream());
            logger.debug("Found config file {} in hub", configFile);
            return configUri;
        } catch (IOException e) {
            logger.debug(
                    "Failed to get config file {} for model {}, {}",
                    configFile,
                    modelId,
                    e.getMessage());
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
            if (modelConfigUri.toString().endsWith("params.json")) {
                MistralModelConfig mistralConfig =
                        JsonUtils.GSON.fromJson(Utils.toString(is), MistralModelConfig.class);
                return new HuggingFaceModelConfig(mistralConfig);
            }
            return JsonUtils.GSON.fromJson(Utils.toString(is), HuggingFaceModelConfig.class);
        } catch (IOException | JsonSyntaxException e) {
            throw new ModelNotFoundException("Invalid huggingface model id: " + modelId, e);
        }
    }

    private static Path buildTrtLlmArtifacts(
            Properties prop, String modelId, String tpDegree, String ppDegree) throws IOException {
        logger.info("Converting model to TensorRT-LLM artifacts");
        String hash = Utils.hash(modelId + tpDegree);
        String download = Utils.getenv("SERVING_DOWNLOAD_DIR", null);
        Path parent = download == null ? Utils.getCacheDir() : Paths.get(download);
        Path trtLlmRepoDir = parent.resolve("trtllm").resolve(hash);
        if (Files.exists(trtLlmRepoDir)) {
            logger.info("TensorRT-LLM artifacts already converted: {}", trtLlmRepoDir);
            return trtLlmRepoDir;
        }

        Path tempDir = Files.createTempDirectory("trtllm");
        logger.info("Writing temp properties to {}", tempDir.toAbsolutePath());
        try (OutputStream os = Files.newOutputStream(tempDir.resolve("serving.properties"))) {
            prop.store(os, "");
        }

        List<String> cmd =
                Arrays.asList(
                        "python",
                        "/opt/djl/partition/trt_llm_partition.py",
                        "--properties_dir",
                        tempDir.toAbsolutePath().toString(),
                        "--trt_llm_model_repo",
                        trtLlmRepoDir.toString(),
                        "--tensor_parallel_degree",
                        tpDegree,
                        "--pipeline_parallel_degree",
                        ppDegree,
                        "--model_path",
                        modelId);
        boolean success = false;
        try {
            logger.info("Converting model to TensorRT-LLM artifacts: {}", (Object) cmd);
            exec(cmd);
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

    /**
     * This represents the config of huggingface models NLP models as well as the config of
     * diffusers models. The config is different for both, but for now we can leverage a single
     * class since we don't need too much information from the config.
     */
    public static final class HuggingFaceModelConfig {

        @SerializedName("model_type")
        private String modelType;

        @SerializedName("architectures")
        private List<String> configArchitectures;

        @SerializedName("auto_map")
        private Map<String, String> autoMap;

        @SerializedName("_diffusers_version")
        private String diffusersVersion;

        @SerializedName("hidden_size")
        private int hiddenSize;

        @SerializedName("intermediate_size")
        private int intermediateSize;

        @SerializedName("max_position_embeddings")
        private int maxPositionEmbeddings;

        @SerializedName("num_attention_heads")
        private int numAttentionHeads;

        @SerializedName("num_hidden_layers")
        private int numHiddenLayers;

        @SerializedName("num_key_value_heads")
        private int numKeyValueHeads;

        @SerializedName("vocab_size")
        private int vocabSize;

        @SerializedName("peft_type")
        private String peftType;

        private Set<String> allArchitectures;

        HuggingFaceModelConfig(MistralModelConfig mistralModelConfig) {
            this.modelType = "mistral";
            this.configArchitectures = List.of("MistralForCausalLM");
            this.hiddenSize = mistralModelConfig.dim;
            this.intermediateSize = mistralModelConfig.hiddenDim;
            this.numAttentionHeads = mistralModelConfig.nHeads;
            this.numHiddenLayers = mistralModelConfig.nLayers;
            this.numKeyValueHeads = mistralModelConfig.nKvHeads;
            this.vocabSize = mistralModelConfig.vocabSize;
        }

        /**
         * Returns the model type of this HuggingFace model.
         *
         * <p>If the model type is not explicitly set, it returns "stable-diffusion" if the
         * diffusers version is set (i.e. it is a diffusers model), or returns null if not (i.e. it
         * is a transformers model).
         *
         * @return the model type
         */
        public String getModelType() {
            if (modelType == null) {
                return diffusersVersion == null ? null : "stable-diffusion";
            }
            return modelType;
        }

        /**
         * Returns the set of all supported architectures for this model.
         *
         * <p>This function will download the model configuration from the HuggingFace Hub the first
         * time it is called, and then cache the result for future invocations.
         *
         * @return the set of all supported architectures
         */
        public Set<String> getArchitectures() {
            if (allArchitectures == null) {
                determineAllArchitectures();
            }
            return allArchitectures;
        }

        /**
         * Returns the default value for the n_positions model configuration. For models that do not
         * have a pre-defined value for n_positions, this function returns the minimum of
         * max_position_embeddings and 4096. If both max_position_embeddings and 4096 are not
         * available, this function returns 0.
         *
         * @return the default value for n_positions
         */
        public int getDefaultNPositions() {
            return Math.min(maxPositionEmbeddings, 4096);
        }

        /**
         * Calculates the number of parameters in a model that is similar to LLaMA. This function
         * takes into account the hidden size, intermediate size, maximum position embeddings,
         * number of hidden layers, vocabulary size, and number of attention heads and key-value
         * heads to calculate the total parameter count.
         *
         * @return the total parameter count for the model
         */
        private long getLlamaLikeParameterCount() {
            long headDim = (long) numAttentionHeads * numKeyValueHeads;
            long embeddings = (long) vocabSize * hiddenSize;
            long qkvProjection = headDim * hiddenSize * numKeyValueHeads * 3;
            long oProjection = (long) hiddenSize * hiddenSize;
            long gateProjection = (long) hiddenSize * intermediateSize * 3;
            return embeddings
                    + numHiddenLayers
                            * (qkvProjection
                                    + oProjection
                                    + gateProjection
                                    + hiddenSize
                                    + hiddenSize)
                    + hiddenSize
                    + embeddings;
        }

        /**
         * Calculates the default parameter count for a model (GPT-2-like).
         *
         * <p>This function takes into account the hidden size, maximum position embeddings, number
         * of hidden layers, vocabulary size, and number of attention heads to calculate the total
         * parameter count.
         *
         * @return the total parameter count for the model
         */
        private long getDefaultParameterCount() {
            long embeddingLayerTotal = (long) (vocabSize + maxPositionEmbeddings) * hiddenSize;
            long attentionTotal = 4L * hiddenSize * hiddenSize;
            long feedForwardTotal = 8L * hiddenSize * hiddenSize;
            long layerNormTotal = 4L * hiddenSize;
            long transformerBlockTotal =
                    (attentionTotal + feedForwardTotal + layerNormTotal) * numHiddenLayers;
            long finalLayerTotal = (long) hiddenSize * vocabSize;
            return embeddingLayerTotal + transformerBlockTotal + finalLayerTotal;
        }

        /**
         * Calculates the total parameter count for the model.
         *
         * @return the total parameter count for the model
         */
        public long getModelParameters() {
            if ("llama".equals(modelType) || "mistral".equals(modelType)) {
                return getLlamaLikeParameterCount();
            }
            return getDefaultParameterCount();
        }

        /**
         * Returns the memory required to store a single batch of sequence data.
         *
         * <p>The memory required is calculated as the product of the sequence length, hidden size,
         * number of hidden layers, and weight in bytes.
         *
         * @param sequenceLength the length in tokens of the sequence
         * @param weightBytes the weight in bytes
         * @return the memory required to store a single batch of sequence data
         */
        public long getApproxMemoryForSingleSequence(int sequenceLength, int weightBytes) {
            return (long) sequenceLength * hiddenSize * numHiddenLayers * weightBytes;
        }

        /**
         * Returns true if the huggingface model id points to a Peft/Lora config.
         *
         * @return whether the huggingface model id points to Peft/Lora model artifacts.
         */
        public boolean isPeftModel() {
            // TODO: refactor and make this better
            // Peft Configs are very different than regular configs and ideally shouldn't be clubbed
            // into this class.
            // This method works now, as the only info we need for the peft model is whether it is
            // peft
            return peftType != null;
        }

        /**
         * Determines all architectures by combining the configured architectures and the auto-map.
         */
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

    /**
     * This represents a Mistral Model Config. Mistral artifacts are different from HuggingFace
     * artifacts. Some Mistral vended models only come in Mistral artifact form.
     */
    static final class MistralModelConfig {

        @SerializedName("dim")
        private int dim;

        @SerializedName("n_layers")
        private int nLayers;

        @SerializedName("head_dim")
        private int headDim;

        @SerializedName("hidden_dim")
        private int hiddenDim;

        @SerializedName("n_heads")
        private int nHeads;

        @SerializedName("n_kv_heads")
        private int nKvHeads;

        @SerializedName("vocab_size")
        private int vocabSize;

        @SerializedName("vision_encoder")
        private Map<String, String> visionEncoder;
    }
}
