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

import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Metrics;
import ai.djl.metric.Unit;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;
import ai.djl.util.RandomUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

class RollingBatch implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(RollingBatch.class);
    private static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private static ExecutorService threadPool =
            Executors.newCachedThreadPool(
                    r -> {
                        Thread t = Executors.defaultThreadFactory().newThread(r);
                        t.setDaemon(true);
                        return t;
                    });

    private PyProcess process;
    private int maxRollingBatchSize;
    private int timeout;
    private boolean stop;
    private List<Request> list;
    private Thread currentThread;
    private ReentrantLock lock;
    private Condition canAdd;
    private Condition canRead;
    private boolean resetRollingBatch;
    private Metrics metrics;
    private Dimension dimension;

    RollingBatch(PyProcess process, Model model, int timeout) {
        this.process = process;
        this.timeout = timeout;
        this.dimension = new Dimension("Model", model.getProperty("metric_dimension", "model"));
        maxRollingBatchSize = model.intProperty("max_rolling_batch_size", 32);
        // TODO: find a way to support custom output_formatter
        list = new ArrayList<>(3);
        lock = new ReentrantLock(true);
        canAdd = lock.newCondition();
        canRead = lock.newCondition();
        threadPool.submit(this);
        if (Boolean.parseBoolean(model.getProperty("log_request_metric"))) {
            int metricsAggregation = model.intProperty("metrics_aggregation", 1000);
            metrics = new Metrics();
            metrics.setLimit(metricsAggregation);
            metrics.setOnLimit(
                    (m, s) -> {
                        MODEL_METRIC.info("{}", m.percentile(s, 50));
                        MODEL_METRIC.info("{}", m.percentile(s, 90));
                    });
        }
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        currentThread = Thread.currentThread();
        while (!stop) {
            int size;
            Input batch = new Input();
            try {
                lock.lock();
                if (list.isEmpty()) {
                    canRead.await();
                }

                if (resetRollingBatch) {
                    batch.addProperty("reset_rollingbatch", "true");
                    resetRollingBatch = false;
                }
                size = list.size();
                for (int i = 0; i < size; ++i) {
                    Request req = list.get(i);
                    // TODO: max 999 batch size
                    String prefix;
                    if (i > 99) {
                        prefix = "batch_" + i + '.';
                    } else if (i > 9) {
                        prefix = "batch_0" + i + '.';
                    } else {
                        prefix = "batch_00" + i + '.';
                    }
                    for (Map.Entry<String, String> entry : req.getProperties()) {
                        String key = prefix + entry.getKey();
                        batch.addProperty(key, entry.getValue());
                    }

                    batch.add(prefix + "data", req.getRequest());
                    String seed = req.getSeed();
                    if (seed != null) {
                        batch.add(prefix + "seed", req.seed);
                    }
                }
                batch.addProperty("batch_size", String.valueOf(size));
            } catch (InterruptedException e) {
                logger.warn("rolling batch loop interrupted.", e);
                break;
            } finally {
                lock.unlock();
            }

            Output output;
            try {
                output = process.predictStandard(batch, timeout, false);
            } catch (EngineException e) {
                logger.warn("prediction failed.", e);
                list.clear();
                resetRollingBatch = true;
                continue;
            }

            try {
                lock.lock();
                PairList<String, BytesSupplier> content = output.getContent();
                // TODO: optimize for conditional killing
                int code = output.getCode();
                Map<String, String> prop = output.getProperties();
                if (code != 200 || content.size() != size) {
                    if (code != 200) {
                        logger.warn("Batch inference failed: {}", output.getMessage());
                    } else {
                        logger.error(
                                "Batch output size mismatch, expected: {}, actual: {}",
                                size,
                                content.size());
                    }
                    Output out = new Output(output.getCode(), "Batch inference failed");
                    BytesSupplier err = BytesSupplier.wrap(JsonUtils.GSON.toJson(out));
                    for (Request req : list) {
                        req.last = true;

                        // Note: This will only change the HTTP response code if the first chunk
                        // has not yet been sent
                        req.output.setCode(code);

                        req.data.appendContent(err, true);
                    }
                    list.clear();
                    resetRollingBatch = true;
                    canAdd.signal();
                    continue;
                }
                Iterator<Map.Entry<String, String>> it = prop.entrySet().iterator();
                Map<Integer, Map<String, String>> map = new ConcurrentHashMap<>();
                while (it.hasNext()) {
                    Map.Entry<String, String> entry = it.next();
                    String key = entry.getKey();
                    if (key.startsWith("batch_")) {
                        it.remove();
                        int pos = key.indexOf('_', 7);
                        if (pos > 0) {
                            int index = Integer.parseInt(key.substring(6, pos));
                            Map<String, String> p =
                                    map.computeIfAbsent(index, i -> new ConcurrentHashMap<>());
                            p.put(key.substring(pos + 1), entry.getValue());
                        }
                    }
                }

                for (int i = 0; i < size; ++i) {
                    Request status = list.get(i);
                    byte[] resp = content.get(i).getValue().getAsBytes();
                    Map<String, String> properties = map.get(i);
                    status.addResponse(resp, properties);
                }
                if (list.removeIf(status -> status.last) || list.size() < maxRollingBatchSize) {
                    canAdd.signal();
                }
                logger.trace("rolling batch size: {}", size);
                if (metrics != null) {
                    metrics.addMetric("RollingBatchSize", size, Unit.COUNT_PER_ITEM, dimension);
                }
            } finally {
                lock.unlock();
            }
        }
    }

    public Output addInput(Input input, int timeout) throws TranslateException {
        try {
            lock.lock();
            if (list.size() >= maxRollingBatchSize) {
                // Input always reflects a single request here
                String requestId = input.getProperty("requestId", "");
                String requestIdLogPrefix = "RequestId=[" + requestId + "]";
                logger.debug(
                        "{} exceed max_rolling_batch_size: {}",
                        requestIdLogPrefix,
                        maxRollingBatchSize);
                if (!canAdd.await(timeout, TimeUnit.SECONDS)) {
                    Metric metric =
                            new Metric("RollingBatchTimeout", list.size(), Unit.COUNT, dimension);
                    MODEL_METRIC.info("{}", metric);
                    logger.warn(
                            "{} Batch is full, cannot process request at this time",
                            requestIdLogPrefix);
                    throw new TranslateException("Time out in: " + timeout);
                }
            }
            String seed = String.valueOf(RandomUtils.nextInt());
            Request req = new Request(input, seed, metrics, dimension);
            list.add(req);
            canRead.signal();
            return req.output;
        } catch (InterruptedException e) {
            throw new TranslateException("Interrupted", e);
        } finally {
            lock.unlock();
        }
    }

    public void shutdown() {
        this.stop = true;
        threadPool.shutdown();
        currentThread.interrupt();
    }
}
