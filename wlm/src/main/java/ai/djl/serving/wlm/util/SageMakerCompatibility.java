/*
 * Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.wlm.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Properties;

/**
 * Utility class for SageMaker backwards compatibility with XGBoost and SKLearn containers.
 * Translates SageMaker environment variables to DJL equivalents.
 */
public final class SageMakerCompatibility {

    private static final Logger logger = LoggerFactory.getLogger(SageMakerCompatibility.class);

    private SageMakerCompatibility() {}

    /**
     * Applies SageMaker compatibility for server-level configurations. Called during ConfigManager
     * initialization.
     *
     * @param properties the properties to modify
     */
    public static void applyServerCompatibility(Properties properties) {
        String maxRequestSize = System.getenv("SAGEMAKER_MAX_REQUEST_SIZE");
        String maxPayloadMb = System.getenv("SAGEMAKER_MAX_PAYLOAD_IN_MB");

        // SAGEMAKER_MAX_REQUEST_SIZE takes precedence over SAGEMAKER_MAX_PAYLOAD_IN_MB
        if (maxRequestSize != null) {
            logger.info(
                    "SageMaker compatibility - translating SAGEMAKER_MAX_REQUEST_SIZE={} to"
                            + " max_request_size",
                    maxRequestSize);
            properties.setProperty("max_request_size", maxRequestSize);

            if (maxPayloadMb != null) {
                logger.warn(
                        "Both SAGEMAKER_MAX_REQUEST_SIZE and SAGEMAKER_MAX_PAYLOAD_IN_MB are set."
                                + " Using SAGEMAKER_MAX_REQUEST_SIZE={} and ignoring"
                                + " SAGEMAKER_MAX_PAYLOAD_IN_MB={}",
                        maxRequestSize,
                        maxPayloadMb);
            }
        } else if (maxPayloadMb != null) {
            try {
                long payloadBytes = Long.parseLong(maxPayloadMb) * 1024 * 1024;
                logger.info(
                        "SageMaker compatibility - translating SAGEMAKER_MAX_PAYLOAD_IN_MB={} to"
                                + " max_request_size={} bytes",
                        maxPayloadMb,
                        payloadBytes);
                properties.setProperty("max_request_size", String.valueOf(payloadBytes));
            } catch (NumberFormatException e) {
                logger.warn("Invalid SAGEMAKER_MAX_PAYLOAD_IN_MB value: {}", maxPayloadMb);
            }
        }
    }

    /**
     * Applies SageMaker compatibility for model-level configurations. Called during
     * ModelInfo.loadServingProperties().
     *
     * @param properties the properties to modify
     */
    public static void applyModelCompatibility(Properties properties) {
        String numWorkers = System.getenv("SAGEMAKER_NUM_MODEL_WORKERS");
        if (numWorkers != null) {
            logger.info(
                    "SageMaker compatibility - translating SAGEMAKER_NUM_MODEL_WORKERS={} to"
                            + " minWorkers/maxWorkers",
                    numWorkers);
            properties.setProperty("minWorkers", numWorkers);
            properties.setProperty("maxWorkers", numWorkers);
        }

        String startupTimeout = System.getenv("SAGEMAKER_STARTUP_TIMEOUT");
        if (startupTimeout != null) {
            logger.info(
                    "SageMaker compatibility - translating SAGEMAKER_STARTUP_TIMEOUT={} to"
                            + " model_loading_timeout",
                    startupTimeout);
            properties.setProperty("option.model_loading_timeout", startupTimeout);
        }

        String predictTimeout = System.getenv("SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS");
        if (predictTimeout != null) {
            logger.info(
                    "SageMaker compatibility - translating"
                            + " SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS={} to predict_timeout",
                    predictTimeout);
            properties.setProperty("option.predict_timeout", predictTimeout);
        }

        String defaultAccept = System.getenv("SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT");
        if (defaultAccept != null) {
            String entryPoint = properties.getProperty("option.entryPoint");
            if (isSklearnOrXgboostHandler(entryPoint)) {
                logger.info(
                        "SageMaker compatibility - setting default accept type to {}",
                        defaultAccept);
                properties.setProperty("option.default_accept", defaultAccept);
            }
        }
    }

    private static boolean isSklearnOrXgboostHandler(String entryPoint) {
        return entryPoint != null
                && ("djl_python.sklearn_handler".equals(entryPoint)
                        || "djl_python.xgboost_handler".equals(entryPoint));
    }
}
