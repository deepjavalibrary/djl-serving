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
package ai.djl.python.engine;

import ai.djl.engine.EngineException;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;

final class TrtLlmUtils {

    private static final Logger logger = LoggerFactory.getLogger(TrtLlmUtils.class);

    private TrtLlmUtils() {}

    static Optional<Path> initTrtLlmModel(PyModel model) throws IOException {
        // check if downloadS3Dir or local model path is a trt-llm repo
        boolean isTrtLlmRepo = isValidTrtLlmModelRepo(model);
        if (!isTrtLlmRepo) {
            return Optional.of(buildTrtLlmArtifacts(model));
        }
        return Optional.empty();
    }

    static Path buildTrtLlmArtifacts(PyModel model) throws IOException {
        logger.info("Converting model to TensorRT-LLM artifacts");
        Path trtLlmRepoDir = Paths.get("/tmp/tensorrtllm");
        String modelId = model.getProperty("model_id");
        // invoke trt-llm build script
        List<String> commandList = getStrings(model, trtLlmRepoDir, modelId);
        try {
            Process exec = new ProcessBuilder(commandList).redirectErrorStream(true).start();
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
            logger.info("TensorRT-LLM artifacts built successfully");
            return trtLlmRepoDir;
        } catch (InterruptedException e) {
            throw new IOException("Failed to build TensorRT-LLM artifacts", e);
        }
    }

    private static List<String> getStrings(PyModel model, Path trtLlmRepoDir, String modelId) {
        List<String> commandList = new ArrayList<>();
        commandList.add("python");
        commandList.add("/opt/djl/partition/trt_llm_partition.py");
        commandList.add("--properties_dir");
        commandList.add(model.getModelPath().toAbsolutePath().toString());
        commandList.add("--trt_llm_model_repo");
        commandList.add(trtLlmRepoDir.toAbsolutePath().toString());
        commandList.add("--gpu_count");
        commandList.add(String.valueOf(CudaUtils.getGpuCount()));
        if (modelId != null) {
            commandList.add("--model_path");
            commandList.add(modelId);
        }
        return commandList;
    }

    static boolean isValidTrtLlmModelRepo(PyModel model) throws IOException {
        Optional<Path> dirToCheckOptional = Optional.empty();
        Path modelPath = Paths.get(model.getProperty("model_id"));
        if (Files.exists(modelPath)) {
            dirToCheckOptional = Optional.of(modelPath);
        }
        if (!dirToCheckOptional.isPresent()) {
            return false;
        }

        Path dirToCheck = dirToCheckOptional.get();
        List<Path> configFiles = new ArrayList<>();
        List<Path> tokenizerFiles = new ArrayList<>();
        try (Stream<Path> walk = Files.walk(dirToCheck)) {
            walk.filter(Files::isRegularFile)
                    .forEach(
                            path -> {
                                if ("config.pbtxt".equals(path.getFileName().toString())) {
                                    // check depth of config.pbtxt
                                    Path relativePath = dirToCheck.relativize(path);
                                    if (relativePath.getNameCount() == 2) {
                                        configFiles.add(path);
                                    }
                                }
                                // TODO: research required tokenizer files and add a tighter check
                                if ("tokenizer_config.json".equals(path.getFileName().toString())) {
                                    tokenizerFiles.add(path);
                                }
                            });
        }
        boolean isValidRepo = !configFiles.isEmpty() && tokenizerFiles.size() == 1;
        if (isValidRepo) {
            logger.info("Valid TRT-LLM model repo found");
        }
        return isValidRepo;
    }
}
