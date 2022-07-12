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
import ai.djl.serving.wlm.Job;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkLoadManager;
import ai.djl.serving.wlm.WorkLoadManager.WorkerPool;
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
    public float[] predict(Arguments arguments, Metrics metrics, int iteration) {

        MemoryTrainingListener.collectMemoryInfo(metrics); // Measure memory before loading model

        Engine engine = Engine.getEngine(arguments.getEngine());
        Device[] devices = engine.getDevices(arguments.getMaxGpus());
        int numOfWorkers = arguments.getThreads();
        int neuronCores = arguments.getNeuronCores();
        if (neuronCores > 0) {
            devices = new Device[neuronCores];
            Arrays.fill(devices, Device.cpu());
            if (numOfWorkers > 1) {
                numOfWorkers = 2 * neuronCores;
            }
        }

        int delay = arguments.getDelay();
        logger.info("WorkLoad Manager inference with {} workers.", numOfWorkers);

        try (WorkLoadManager wlm = new WorkLoadManager();
                ModelInfo<Void, float[]> modelInfo =
                        new ModelInfo<>("model", loadModelCriteria(arguments, devices[0]))) {

            WorkerPool<Void, float[]> wp = wlm.registerModel(modelInfo);
            int workersPerDevice = numOfWorkers / devices.length;
            for (Device device : devices) {
                wp.scaleWorkers(device, workersPerDevice, workersPerDevice);
            }

            MemoryTrainingListener.collectMemoryInfo(
                    metrics); // Measure memory before worker kickoff

            metrics.addMetric("start", System.currentTimeMillis(), Unit.MILLISECONDS);
            CompletableFuture<float[]>[] results = new CompletableFuture[iteration];
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

            metrics.addMetric("end", System.currentTimeMillis(), Unit.MILLISECONDS);

            long successsfulResults =
                    Arrays.stream(results).mapToInt(f -> f.join() != null ? 1 : 0).count();

            if (successsfulResults != iteration) {
                logger.error(
                        "Only {}/{} results successfully finished.", successsfulResults, iteration);
                return null;
            }

            return results[0].join();
        }
    }

    @Override
    protected boolean benchmarkUsesThreads() {
        return true;
    }
}
