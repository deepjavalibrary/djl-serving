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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.yaml.snakeyaml.Yaml;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Properties;

/** A utility class to detect optimal engine for SageMaker saved model. */
public final class SageMakerUtils {

    private static final Logger logger = LoggerFactory.getLogger(SageMakerUtils.class);

    private static final String FRAMEWORK = "Framework";
    private static final String FRAMEWORK_VERSION = "FrameworkVersion";
    private static final String INFERENCE_SPEC = "InferenceSpec";
    private static final String INFERENCE_SPEC_HMAC = "InferenceSpecHMAC";
    private static final String METADATA = "metadata.yaml";
    private static final String MODEL = "Model";
    private static final String MODEL_TYPE = "ModelType";
    private static final String PYTORCH_MODEL_TYPE = "PyTorchModel";
    private static final String SCHEMA = "Schema";
    private static final String SCHEMA_HMAC = "SchemaHMAC";
    private static final String TASK = "Task";
    private static final String VERSION = "Version";
    private static final String XGBOOST_MODEL_TYPE = "XGBoostModel";

    private SageMakerUtils() {}

    static String inferSageMakerEngine(ModelInfo<?, ?> modelInfo) throws ModelException {
        Properties prop = modelInfo.prop;
        Path modelDir = modelInfo.modelDir;
        Path metaDataFile = modelDir.resolve(METADATA);

        // Load metadata information from metadata.yaml
        Map<String, Object> metaDataMap;
        Yaml metadata = new Yaml();
        try (InputStream inputStream = Files.newInputStream(metaDataFile.toAbsolutePath())) {
            metaDataMap = metadata.load(inputStream);
            logger.info("Successfully loaded metadata.yaml");
        } catch (IOException fileNotFoundException) {
            logger.error("Cannot find valid metadata.yaml: {}", metaDataFile);
            throw new ModelNotFoundException(
                    "Invalid metadata destination: " + metaDataFile, fileNotFoundException);
        }

        if (!validateMetaData(metaDataMap)) {
            // If metadata is not a valid format, try native DJL serving infer engine.
            return null;
        }

        // To validate both schema and inferenceSpec
        boolean customizedSchema = hasCustomizedSchema(metaDataMap);
        boolean customizedInferenceSpec = hasCustomizedInferenceSpec(metaDataMap);
        if (customizedSchema || customizedInferenceSpec) {
            // For either customized schema or customized inference spec using python engine with
            // sagemaker entry point
            logger.info(
                    "Customized schema builder or inference spec is detected, using python"
                            + " backend");
            prop.setProperty("option.entryPoint", "djl_python.sagemaker");
            return "Python";
        }

        if (XGBOOST_MODEL_TYPE.equals(metaDataMap.get(MODEL_TYPE))) {
            prop.setProperty("option.modelName", metaDataMap.get(MODEL).toString());
            return "XGBoost";
        } else if (PYTORCH_MODEL_TYPE.equals(metaDataMap.get(MODEL_TYPE))) {
            prop.setProperty("option.modelName", metaDataMap.get(MODEL).toString());
            return "PyTorch";
        } else {
            logger.error("Invalid model type: " + metaDataMap.get(MODEL_TYPE));
            throw new ModelException(
                    String.format(
                            "Model type %s is not supported in SageMaker model format yet",
                            metaDataMap.get(MODEL_TYPE).toString()));
        }
    }

    private static boolean validateMetaData(Map<String, Object> metaDataMap) {
        // Required field for metadata in SageMaker model format
        return metaDataMap.containsKey(FRAMEWORK)
                && metaDataMap.containsKey(FRAMEWORK_VERSION)
                && metaDataMap.containsKey(MODEL)
                && metaDataMap.containsKey(MODEL_TYPE)
                && metaDataMap.containsKey(VERSION)
                && metaDataMap.containsKey(TASK);
    }

    private static boolean hasCustomizedSchema(Map<String, Object> metaDataMap)
            throws ModelException {
        if (metaDataMap.containsKey(SCHEMA)) {
            if (!metaDataMap.containsKey(SCHEMA_HMAC)) {
                throw new ModelException(
                        "Invalid SageMaker Model format due to SchemaHMAC is not found");
            }
            return true;
        }
        return false;
    }

    private static boolean hasCustomizedInferenceSpec(Map<String, Object> metaDataMap)
            throws ModelException {
        if (metaDataMap.containsKey(INFERENCE_SPEC)) {
            if (!metaDataMap.containsKey(INFERENCE_SPEC_HMAC)) {
                throw new ModelException(
                        "Invalid SageMaker Model format due to InferenceSpecHMAC is not found");
            }
            return true;
        }
        return false;
    }
}
