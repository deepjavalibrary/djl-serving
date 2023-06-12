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

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.PairList;

import com.google.gson.JsonObject;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.ReentrantLock;

class RollingBatch implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger(RollingBatch.class);

    private static ExecutorService threadPool = Executors.newCachedThreadPool();

    private PyProcess process;
    private int maxRollingBatchSize;
    private int timeout;
    private boolean stop;
    private List<Request> list;
    private Thread currentThread;
    private ReentrantLock lock;
    private Condition canAdd;
    private Condition canRead;

    RollingBatch(PyProcess process, int maxRollingBatchSize, int timeout) {
        this.process = process;
        this.maxRollingBatchSize = maxRollingBatchSize;
        this.timeout = timeout;
        list = new ArrayList<>(3);
        lock = new ReentrantLock();
        canAdd = lock.newCondition();
        canRead = lock.newCondition();
        threadPool.submit(this);
    }

    /** {@inheritDoc} */
    @Override
    public void run() {
        currentThread = Thread.currentThread();
        while (!stop) {
            try {
                Thread.yield();
                lock.lock();
                if (list.isEmpty()) {
                    canRead.await();
                }

                Input batch = new Input();
                int size = list.size();
                batch.addProperty("batch_size", String.valueOf(size));
                for (int i = 0; i < size; ++i) {
                    Request req = list.get(i);
                    String prefix = "batch_" + i + ".data";
                    batch.add(prefix, req.getRequest());
                }

                // TODO: Handler error case

                Output output = process.predict(batch, timeout, false);
                PairList<String, BytesSupplier> content = output.getContent();
                if (content.size() != size) {
                    throw new TranslateException(
                            "Batch output size mismatch, expected: "
                                    + size
                                    + ", actual: "
                                    + content.size());
                }
                for (int i = 0; i < size; ++i) {
                    Request status = list.get(i);
                    String json = content.get(i).getValue().getAsString();
                    status.addResponse(json);
                }
                list.removeIf(status -> status.last);
                if (list.size() < maxRollingBatchSize) {
                    canAdd.signal();
                }
            } catch (InterruptedException e) {
                break;
            } catch (TranslateException e) {
                logger.error("RollingBatch thread died, killing python process.", e);
                process.stopPythonProcess();
            } finally {
                lock.unlock();
            }
        }
    }

    public Output addInput(Input input, int timeout) throws TranslateException {
        try {
            lock.lock();
            if (list.size() >= maxRollingBatchSize) {
                if (!canAdd.await(timeout, TimeUnit.SECONDS)) {
                    throw new TranslateException("Time out in: " + timeout);
                }
            }
            Request req = new Request(input);
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

    private static final class Request {

        Input input;
        ChunkedBytesSupplier data;
        Output output;
        String nextToken;
        boolean last;

        Request(Input input) {
            this.input = input;
            data = new ChunkedBytesSupplier();
            output = new Output();
            output.add(data);
        }

        BytesSupplier getRequest() {
            if (nextToken != null) {
                return BytesSupplier.wrap("{\"inputs\": [\"" + nextToken + "\"]}");
            }
            return input.getData();
        }

        void addResponse(String json) {
            JsonObject element = JsonUtils.GSON.fromJson(json, JsonObject.class);
            last = element.get("last").getAsBoolean();
            nextToken = element.get("data").getAsString();
            data.appendContent(BytesSupplier.wrap(nextToken), last);
        }
    }
}
