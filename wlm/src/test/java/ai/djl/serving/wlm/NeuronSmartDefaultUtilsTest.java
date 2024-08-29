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
import ai.djl.util.JsonUtils;
import ai.djl.util.NeuronUtils;

import org.mockito.MockedStatic;
import org.mockito.Mockito;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Properties;

/**
 * {@link NeuronSmartDefaultUtils}.
 *
 * @author tyoster @ Amazon.com, Inc.
 */
public class NeuronSmartDefaultUtilsTest {

    // Known model parameter tests for LMI Utils HuggingFaceModelConfig
    @Test
    public void testModelConfigParametersLlama() throws IOException {
        LmiUtils.HuggingFaceModelConfig modelConfig = get70BLlamaHuggingFaceModelConfig();
        Assert.assertEquals(modelConfig.getDefaultNPositions(), 4096);
        Assert.assertEquals(modelConfig.getModelParameters(), 71895883776L);
    }

    @Test
    public void testModelConfigParametersDefault() throws IOException {
        LmiUtils.HuggingFaceModelConfig modelConfig = getDefaultHuggingFaceModelConfig();
        Assert.assertEquals(modelConfig.getDefaultNPositions(), 1);
        Assert.assertEquals(modelConfig.getModelParameters(), 19L);
    }

    @Test
    public void testModelConfigParametersNoParameters() throws IOException {
        LmiUtils.HuggingFaceModelConfig modelConfig = getNoParametersHuggingFaceModelConfig();
        Assert.assertEquals(modelConfig.getDefaultNPositions(), 1);
        Assert.assertEquals(modelConfig.getModelParameters(), 0L);
    }

    // Standard use tests on without Neuron device available
    @Test
    public void testApplySmartDefaults70BModel() throws IOException {
        Properties prop = new Properties();
        LmiUtils.HuggingFaceModelConfig modelConfig = get70BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "4096");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertFalse(prop.containsKey("option.max_rolling_batch_size"));
    }

    @Test
    public void testApplySmartDefaultsQuantize8BModel() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.quantize", "static_int8");
        LmiUtils.HuggingFaceModelConfig modelConfig = get8BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "4096");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "8");
    }

    @Test
    public void testApplySmartDefaults2BModel() throws IOException {
        Properties prop = new Properties();
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "2048");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "64");
    }

    @Test
    public void testApplySmartDefaultsQuantize2BModel() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.quantize", "static_int8");
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "2048");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "128");
    }

    @Test
    public void testApplySmartDefaultsWithNPositions() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.n_positions", "128");
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "128");
    }

    @Test
    public void testApplySmartDefaultsWithTPDegree() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.tensor_parallel_degree", "1");
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "2048");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "64");
    }

    @Test
    public void testApplySmartDefaultsWithMaxRollingBatch() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.max_rolling_batch_size", "64");
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "2048");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
    }

    @Test
    public void testApplySmartDefaultsWithTPMax() throws IOException {
        Properties prop = new Properties();
        prop.setProperty("option.tensor_parallel_degree", "max");
        LmiUtils.HuggingFaceModelConfig modelConfig = get2BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(false);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "2048");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "1");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "64");
    }

    @Test
    public void testApplySmartDefaultsWithNeuron() throws IOException {
        Properties prop = new Properties();
        LmiUtils.HuggingFaceModelConfig modelConfig = get70BLlamaHuggingFaceModelConfig();
        try (MockedStatic<NeuronUtils> mockedStatic = Mockito.mockStatic(NeuronUtils.class)) {
            mockedStatic.when(NeuronUtils::hasNeuron).thenReturn(true);
            mockedStatic.when(NeuronUtils::getNeuronCores).thenReturn(32);
            NeuronSmartDefaultUtils smartDefaultUtils = new NeuronSmartDefaultUtils();
            smartDefaultUtils.applySmartDefaults(prop, modelConfig);
        }
        Assert.assertEquals(prop.getProperty("option.n_positions"), "4096");
        Assert.assertEquals(prop.getProperty("option.tensor_parallel_degree"), "32");
        Assert.assertEquals(prop.getProperty("option.max_rolling_batch_size"), "32");
    }

    // Helper methods
    public HuggingFaceModelConfig get2BLlamaHuggingFaceModelConfig() throws IOException {
        try (Reader reader =
                Files.newBufferedReader(
                        Paths.get("src/test/resources/smart-default-model/2b/config.json"))) {
            return JsonUtils.GSON.fromJson(reader, HuggingFaceModelConfig.class);
        }
    }

    public HuggingFaceModelConfig get8BLlamaHuggingFaceModelConfig() throws IOException {
        try (Reader reader =
                Files.newBufferedReader(
                        Paths.get("src/test/resources/smart-default-model/8b/config.json"))) {
            return JsonUtils.GSON.fromJson(reader, HuggingFaceModelConfig.class);
        }
    }

    public HuggingFaceModelConfig get70BLlamaHuggingFaceModelConfig() throws IOException {
        try (Reader reader =
                Files.newBufferedReader(
                        Paths.get("src/test/resources/smart-default-model/70b/config.json"))) {
            return JsonUtils.GSON.fromJson(reader, HuggingFaceModelConfig.class);
        }
    }

    public HuggingFaceModelConfig getDefaultHuggingFaceModelConfig() throws IOException {
        try (Reader reader =
                Files.newBufferedReader(
                        Paths.get("src/test/resources/smart-default-model/unit/config.json"))) {
            return JsonUtils.GSON.fromJson(reader, HuggingFaceModelConfig.class);
        }
    }

    public HuggingFaceModelConfig getNoParametersHuggingFaceModelConfig() throws IOException {
        try (Reader reader =
                Files.newBufferedReader(
                        Paths.get("src/test/resources/smart-default-model/empty/config.json"))) {
            return JsonUtils.GSON.fromJson(reader, HuggingFaceModelConfig.class);
        }
    }
}
