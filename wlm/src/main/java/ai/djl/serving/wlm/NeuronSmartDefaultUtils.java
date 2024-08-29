/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.serving.wlm.LmiUtils.HuggingFaceModelConfig;
import ai.djl.util.NeuronUtils;

import java.util.ArrayList;
import java.util.List;
import java.util.Properties;

/** A utility class to auto configure LMI Neuron model properties. */
public class NeuronSmartDefaultUtils {

    private static final float BILLION = 1_000_000_000.0F;
    private static final int MAX_ROLLING_BATCH = 128; // Current cap for NeuronSDK 2.19.1
    private static final float MEMORY_PER_CORE =
            16.0F; // Currently there is only one config w/ 16 gb per core

    private int availableCores;
    private float modelSizeInGb;
    private float sequenceSizeInGb;

    /**
     * Applies smart defaults for Neuron models.
     *
     * <p>This method sets the following properties if not already set:
     *
     * <ul>
     *   <li>option.n_positions: The default n_positions for the model.
     *   <li>option.tensor_parallel_degree: A heuristic based on available memory.
     *   <li>option.max_rolling_batch_size: A heuristic based on available memory.
     * </ul>
     *
     * @param prop the properties to update
     * @param modelConfig the model configuration to use
     */
    public void applySmartDefaults(Properties prop, HuggingFaceModelConfig modelConfig) {
        if (!prop.containsKey("option.n_positions")) {
            if (modelConfig.getDefaultNPositions() <= 0) {
                // If n positions cannot be determined skip smart defaults
                return;
            }
            prop.setProperty(
                    "option.n_positions", String.valueOf(modelConfig.getDefaultNPositions()));
        }
        setInternalSettings(prop, modelConfig);
        setHeuristicNeuronTPDegree(prop);
        setHeuristicNeuronMaxRollingBatch(prop);
    }

    /**
     * Sets the internal settings for the NeuronSmartDefaultUtils instance.
     *
     * @param prop the properties to retrieve settings from
     * @param modelConfig the model configuration to use for calculations
     */
    private void setInternalSettings(Properties prop, HuggingFaceModelConfig modelConfig) {
        // Internal settings
        int nPositions = Integer.parseInt(prop.getProperty("option.n_positions", "0"));
        if (NeuronUtils.hasNeuron()) {
            availableCores = NeuronUtils.getNeuronCores();
        } else {
            availableCores = 1;
        }
        int paramBytes = prop.containsKey("option.quantize") ? 1 : 2;
        modelSizeInGb = (paramBytes * modelConfig.getModelParameters()) / BILLION;
        sequenceSizeInGb =
                modelConfig.getApproxMemoryForSingleSequence(nPositions, paramBytes)
                        / (1024.0F * 1024.0F * 1024.0F);
    }

    /**
     * Calculates the adjusted model size in GB, given a tensor parallel degree.
     *
     * <p>This method takes the model size in GB and adjusts it based on the tensor parallel degree.
     * The adjustment is a linear relationship between the model size and the tensor parallel
     * degree. The adjustment is based on the estimated memory increase due to the tensor parallel
     * degree.
     *
     * @param tpDegree the tensor parallel degree
     * @return the adjusted model size in GB
     */
    private float getAdjustedModelSizeInGb(int tpDegree) {
        return modelSizeInGb * (1.0F + ((tpDegree * 2 - 2) / 100.0F));
    }

    /**
     * Sets a heuristic value for tensor parallel degree if not already set in model properties.
     *
     * <p>This method sets the value of tensor parallel degree by iterating through the available
     * core configurations and checks if the current core configuration can support the maximum
     * rolling batch size that can fit in the available memory. If the current configuration can
     * support the maximum rolling batch size, then the current core configuration is used as the
     * tensor parallel degree.
     *
     * <p>If the maximum rolling batch size is not set, then the maximum instance concurrency is
     * used as the maximum rolling batch size.
     *
     * <p>This method is called by the LMI model server when it is starting up and is used to set
     * the tensor parallel degree if it is not already set in the model properties.
     *
     * @param prop the model properties
     */
    private void setHeuristicNeuronTPDegree(Properties prop) {
        int tpDegree = availableCores;
        float totalMemory = tpDegree * MEMORY_PER_CORE;
        // Disambiguate "max" and available cores
        if (prop.containsKey("option.tensor_parallel_degree")
                && "max".equals(prop.getProperty("option.tensor_parallel_degree"))) {
            prop.setProperty("option.tensor_parallel_degree", String.valueOf(availableCores));
            return;
        }

        List<Integer> coreConfigs = availableCoreConfigs();
        if (!prop.containsKey("option.tensor_parallel_degree")
                && !prop.containsKey("option.max_rolling_batch_size")) {
            // Set tensor parallel degree based off of maximizing instance concurrency with variable
            // rolling batch size
            int totalInstanceConcurrency = getMaxConcurrency(totalMemory, tpDegree);
            for (int coreConfig : coreConfigs) {
                float maxMemory = coreConfig * MEMORY_PER_CORE;
                int maxConcurrency = getMaxConcurrency(maxMemory, coreConfig);
                if (maxConcurrency >= totalInstanceConcurrency && coreConfig <= tpDegree) {
                    tpDegree = coreConfig;
                    totalInstanceConcurrency = maxConcurrency;
                }
            }
            prop.setProperty("option.tensor_parallel_degree", String.valueOf(tpDegree));
        } else if (!prop.containsKey("option.tensor_parallel_degree")) {
            // Set tensor parallel degree by minimizing TP degree that supports fixed batch size
            int batchSize = Integer.parseInt(prop.getProperty("option.max_rolling_batch_size"));
            int totalInstanceConcurrency =
                    getMaxConcurrencyWithBatch(totalMemory, tpDegree, batchSize);
            for (int coreConfig : coreConfigs) {
                float maxMemory = coreConfig * MEMORY_PER_CORE;
                int maxConcurrency = getMaxConcurrencyWithBatch(maxMemory, coreConfig, batchSize);
                if (maxConcurrency >= totalInstanceConcurrency && coreConfig <= tpDegree) {
                    tpDegree = coreConfig;
                    totalInstanceConcurrency = maxConcurrency;
                }
            }
            prop.setProperty("option.tensor_parallel_degree", String.valueOf(tpDegree));
        }
    }

    /**
     * Finds the largest power of 2 less than or equal to n.
     *
     * @param n the input number
     * @return the largest power of 2 less than or equal to n
     */
    private int getMaxPowerOf2(int n) {
        n = Math.min(n, MAX_ROLLING_BATCH);
        if (n != 0 && (n & (n - 1)) == 0) {
            return n;
        }
        int maxPowerOf2 = 1;
        while (maxPowerOf2 < n) {
            maxPowerOf2 *= 2;
        }
        return maxPowerOf2 / 2;
    }

    /**
     * Calculates the maximum number of concurrent requests that can be served by a model given the
     * total memory available for the model and the sequence size.
     *
     * <p>The maximum number of concurrent requests is calculated as the largest power of 2 less
     * than or equal to the total memory divided by the sequence size.
     *
     * @param totalMemory the total memory available for the model
     * @param tpDegree the tensor parallel degree
     * @return the maximum number of concurrent requests
     */
    private int getMaxConcurrency(float totalMemory, int tpDegree) {
        int maxConcurrency =
                (int) ((totalMemory - getAdjustedModelSizeInGb(tpDegree)) / sequenceSizeInGb);
        maxConcurrency = getMaxPowerOf2(maxConcurrency);
        return Math.min(maxConcurrency, MAX_ROLLING_BATCH);
    }

    /**
     * Calculates the maximum number of concurrent requests that can be served by a model given the
     * total memory available for the model and the sequence size.
     *
     * @param totalMemory the total memory available for the model
     * @param tpDegree the tensor parallel degree
     * @param batchSize the maximum number of requests that can be processed in a single batch
     * @return the maximum number of concurrent requests that can be served
     */
    private int getMaxConcurrencyWithBatch(float totalMemory, int tpDegree, int batchSize) {
        int maxConcurrency =
                (int) ((totalMemory - getAdjustedModelSizeInGb(tpDegree)) / sequenceSizeInGb);
        maxConcurrency = getMaxPowerOf2(maxConcurrency);
        maxConcurrency = Math.min(maxConcurrency, batchSize);
        if (maxConcurrency == batchSize) {
            return maxConcurrency;
        }
        return 0;
    }

    /**
     * Builds the available core configurations for a given number of cores.
     *
     * <p>The available core configurations are those that are less than or equal to the total
     * number of cores. This method returns a list of available core configurations for the given
     * number of cores.
     *
     * @return the list of available core configurations
     */
    private List<Integer> availableCoreConfigs() {
        List<Integer> coreConfigs = new ArrayList<>();
        List<Integer> availableCoreConfigs = buildCoreConfigs(availableCores);
        int coresPerModel = (int) Math.ceil(modelSizeInGb / MEMORY_PER_CORE);
        for (int coreConfig : availableCoreConfigs) {
            if (coresPerModel >= coreConfig) {
                coreConfigs.add(coreConfig);
            }
        }
        return coreConfigs;
    }

    /**
     * Builds the available core configurations for a given number of cores.
     *
     * <p>The available core configurations are those that are less than or equal to the total
     * number of cores. This method returns a list of available core configurations for the given
     * number of cores.
     *
     * @param nCores the number of cores to build the configurations for
     * @return the list of available core configurations
     */
    private List<Integer> buildCoreConfigs(int nCores) {
        List<Integer> coreConfigs = new ArrayList<>();
        // Add all powers of 2 up to the given number of cores
        for (int i = 1; i <= 8; i *= 2) {
            // Skip TP=4 for nCores=32 as it is not supported
            if (i != 4 || nCores != 32) {
                coreConfigs.add(i);
            }
        }
        // Add the given number of cores to the list
        coreConfigs.add(nCores);
        return coreConfigs;
    }

    /**
     * Sets the max rolling batch size based on the TP degree and the model memory size.
     *
     * <p>If the max rolling batch size is not set, this method sets it to the maximum number of
     * concurrent requests that can be served by a model given the total memory available for the
     * model and the sequence size.
     *
     * @param prop the properties to set the max rolling batch size to
     */
    private void setHeuristicNeuronMaxRollingBatch(Properties prop) {
        int tpDegree =
                Integer.parseInt(
                        prop.getProperty(
                                "option.tensor_parallel_degree", String.valueOf(availableCores)));
        if (!prop.containsKey("option.max_rolling_batch_size")) {
            int maxRollingBatchSize = getMaxConcurrency(tpDegree * MEMORY_PER_CORE, tpDegree);
            if (maxRollingBatchSize > 0) {
                prop.setProperty(
                        "option.max_rolling_batch_size", String.valueOf(maxRollingBatchSize));
            }
        }
    }
}
