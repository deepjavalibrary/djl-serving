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

import ai.djl.ModelException;
import ai.djl.engine.EngineException;
import ai.djl.util.Utils;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TrtLLMUtils {

  private static final Logger logger = LoggerFactory.getLogger(TrtLLMUtils.class);

  public static Optional<Path> initTrtLlmModel(PyModel model) throws ModelException, IOException {
    // check if downloadS3Dir or local model path is a trt-llm repo
    boolean isTrtLlmRepo = isValidTrtLlmModelRepo(model);
    if (!isTrtLlmRepo) {
      return Optional.of(buildTrtLlmArtifacts(model));
    }
    return Optional.empty();
  }

  public static Path buildTrtLlmArtifacts(PyModel model) throws ModelException, IOException {
    logger.info("Converting model to TensorRT-LLM artifacts");
    Path trtLlmRepoDir = Paths.get("/tmp/tensorrtllm");
    String modelId = model.getProperty("model_id");
    // invoke trt-llm build script
    List<String> commandList = new ArrayList<>();
    commandList.add("python");
    commandList.add("/opt/djl/partition/trt_llm_partition.py");
    commandList.add("--properties_dir");
    commandList.add(model.getModelPath().toAbsolutePath().toString());
    commandList.add("--trt_llm_model_repo");
    commandList.add(trtLlmRepoDir.toAbsolutePath().toString());
    if (modelId != null) {
      commandList.add("--model_path");
      commandList.add(modelId);
    }
    try {
      Process exec = new ProcessBuilder(commandList).redirectErrorStream(true).start();
      String logOutput;
      try (InputStream is = exec.getInputStream()) {
        logOutput = Utils.toString(is);
      }
      int exitCode = exec.waitFor();
      if (0 != exitCode || logOutput.startsWith("ERROR ")) {
        logger.error(logOutput);
        throw new EngineException("Download model failed: " + logOutput);
      } else {
        logger.info(logOutput);
      }
    } catch (IOException | InterruptedException e) {
      throw new ModelException("Failed to build TensorRT-LLM artifacts", e);
    }
    logger.info("TensorRT-LLM artifacts built successfully");
    return trtLlmRepoDir;
  }

  public static boolean isValidTrtLlmModelRepo(PyModel model) throws IOException {
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
