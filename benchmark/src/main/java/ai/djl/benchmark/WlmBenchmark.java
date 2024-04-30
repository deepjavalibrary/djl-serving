/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.benchmark;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.metric.Unit;
import ai.djl.repository.zoo.Criteria;
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkerPool;
import ai.djl.training.listener.MemoryTrainingListener;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.concurrent.CompletableFuture;

/** A class that runs a benchmark using the {@link WorkLoadManager}. */
public class WlmBenchmark extends AbstractBenchmark {

    private static final Logger logger = LoggerFactory.getLogger(WlmBenchmark.class);

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    public String predict(Arguments arguments, Metrics metrics, int iteration) {
        MemoryTrainingListener.collectMemoryInfo(metrics); // Measure memory before loading model

        Engine engine = Engine.getEngine(arguments.getEngine());
        String[] devices;
        int numOfWorkers = arguments.getThreads();
        int neuronCores = arguments.getNeuronCores();
        if (neuronCores > 0) {
            devices = new String[neuronCores];
            for (int i = 0; i < neuronCores; ++i) {
                devices[i] = "nc" + i;
            }
            if (numOfWorkers > 1) {
                numOfWorkers = 2 * neuronCores;
            }
        } else {
            int gpuCount = engine.getGpuCount();
            if (gpuCount > 0) {
                devices = new String[gpuCount];
                for (int i = 0; i < gpuCount; ++i) {
                    devices[i] = String.valueOf(i);
                }
            } else {
                devices = new String[] {"-1"};
            }
        }

        int delay = arguments.getDelay();
        logger.info("WorkLoad Manager inference with {} workers.", numOfWorkers);

        Device device = Device.fromName(devices[0], engine);
        WorkLoadManager wlm = new WorkLoadManager();
        Criteria<Void, String> criteria = loadModelCriteria(arguments, device);
        ModelInfo<Void, String> modelInfo =
                new ModelInfo<>("model", arguments.getModelUrl(), criteria);

        int workersPerDevice = numOfWorkers / devices.length;
        modelInfo.setMinWorkers(workersPerDevice);
        modelInfo.setMaxWorkers(workersPerDevice);
        WorkerPool<Void, String> wp = wlm.registerWorkerPool(modelInfo);
        for (String deviceName : devices) {
            wp.initWorkers(deviceName);
        }

        // Measure memory before worker kickoff
        MemoryTrainingListener.collectMemoryInfo(metrics);

        metrics.addMetric("start", System.currentTimeMillis(), Unit.MILLISECONDS);
        CompletableFuture<String>[] results = new CompletableFuture[iteration];
        for (int i = 0; i < iteration; i++) {
            try {
                results[i] = wlm.runJob(new Job<>(modelInfo, null));
                if (delay > 0) {
                    Thread.sleep(delay);
                }
            } catch (InterruptedException e) {
                logger.error("", e);
            }
        }

        CompletableFuture.allOf(results).join();
        modelInfo.close();
        wlm.close();

        metrics.addMetric("end", System.currentTimeMillis(), Unit.MILLISECONDS);

        long successfulResults = Arrays.stream(results).count();

        if (successfulResults != iteration) {
            logger.error("Only {}/{} results successfully finished.", successfulResults, iteration);
            return "";
        }

        return results[0].join();
    }

    @Override
    protected boolean benchmarkUsesThreads() {
        return true;
    }
}
