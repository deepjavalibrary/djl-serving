/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.metric.Metric;
import ai.djl.modality.Input;
import ai.djl.modality.Output;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Scanner;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

class PyProcess {

    static final Logger logger = LoggerFactory.getLogger(PyProcess.class);
    static final Logger MODEL_METRIC = LoggerFactory.getLogger("model_metric");

    private PyEnv pyEnv;
    private Model model;
    private int workerId;
    private Process process;
    private String pid;
    private List<Connection> connections;
    private CountDownLatch latch;
    private volatile boolean started; // NOPMD
    private AtomicInteger restartCount;
    private CompletableFuture<Void> restartFuture;
    private boolean trtLlmMode;

    private static AtomicInteger counter = new AtomicInteger(0);

    PyProcess(Model model, PyEnv pyEnv, int workerId) {
        this.model = model;
        this.pyEnv = pyEnv;
        this.workerId = workerId;
        int port = counter.getAndIncrement();
        if (pyEnv.isMpiMode()) {
            int tensorParallelDegree = pyEnv.getTensorParallelDegree();
            connections = new ArrayList<>(tensorParallelDegree);
            for (int i = 0; i < tensorParallelDegree; ++i) {
                connections.add(new Connection(pyEnv, port, i));
            }
            counter.set(port + tensorParallelDegree);
        } else {
            connections = Collections.singletonList(new Connection(pyEnv, port, -1));
        }
        restartCount = new AtomicInteger(0);
        // TODO: avoid using this hack when TRT-LLM improve its behavior
        trtLlmMode = "trtllm".equals(model.getProperty("rolling_batch"));
    }

    Output predict(Input inputs, int timeout, boolean initialLoad) {
        try {
            if (inputs.getProperty("handler", null) == null) {
                String handler = pyEnv.getHandler();
                if (handler != null) {
                    inputs.addProperty("handler", handler);
                }
            }

            List<CompletableFuture<Output>> futures = new ArrayList<>(connections.size());
            if (initialLoad || !trtLlmMode) {
                for (Connection conn : connections) {
                    futures.add(conn.send(inputs));
                }
            } else {
                futures.add(connections.get(0).send(inputs));
            }

            Output output = null;
            if (trtLlmMode) {
                output = futures.get(0).get(timeout, TimeUnit.SECONDS);
            } else {
                for (CompletableFuture<Output> future : futures) {
                    output = future.get(timeout, TimeUnit.SECONDS);
                }
            }

            if (initialLoad && output != null) {
                int code = output.getCode();
                if (code >= 300) {
                    if (code == 507) {
                        throw new EngineException("OOM");
                    }
                    if (pyEnv.isFailOnInitialize()) {
                        throw new EngineException(
                                "Failed to initialize model: " + output.getMessage());
                    }
                    logger.warn("Model doesn't support initialize: {}", output.getMessage());
                } else {
                    logger.info("Model [{}] initialized.", model.getName());
                }
            }

            return output;
        } catch (Throwable e) { // use Throwable to workaround spotbug false alarm
            logger.debug("predict[init={}] exception: {}", initialLoad, e.getClass().getName());
            stopPythonProcess(!initialLoad);
            if (!initialLoad) {
                logger.info("Restart python process ...");
                restartFuture = CompletableFuture.runAsync(this::startPythonProcess);
            }
            if (e instanceof EngineException) {
                throw (EngineException) e;
            }
            throw new EngineException(e);
        }
    }

    synchronized void startPythonProcess() {
        try {
            int id = restartCount.get();
            int port = connections.get(0).getPort();
            logger.info("Start process: {} - retry: {}", port, id);
            pyEnv.installDependency(model.getModelPath());
            process = Connection.startPython(pyEnv, model, workerId, port);
            pid = process.toString().split(", ")[0].replace("Process[pid=", "");

            String modelName = model.getName();
            modelName = modelName.substring(0, Math.min(modelName.length(), 15));
            String threadName = "W-" + pid + '-' + modelName;
            ReaderThread err =
                    new ReaderThread(threadName, process.getErrorStream(), true, this, id);
            ReaderThread out =
                    new ReaderThread(threadName, process.getInputStream(), false, this, id);
            latch = new CountDownLatch(connections.size());
            err.start();
            out.start();
            if (!latch.await(2, TimeUnit.MINUTES)) {
                throw new EngineException("Python process startup time out.");
            }
            if (!started) {
                logger.warn("Process not started, waiting for process end ...");
                int exitCode = process.waitFor();
                throw new IllegalThreadStateException(
                        "Python stream closed unexpectedly, exit code: " + exitCode);
            }

            for (Connection conn : connections) {
                conn.connect();
            }

            // initialize model with an empty request
            Input init = new Input();
            init.setProperties(pyEnv.getInitParameters());
            predict(init, pyEnv.getModelLoadingTimeout(), true);
        } catch (EngineException e) {
            started = false;
            throw e;
        } catch (InterruptedException e) {
            started = false;
            throw new EngineException("Worker startup cancelled.", e);
        } catch (IOException e) {
            started = false;
            throw new EngineException("Failed connect to Python worker process.", e);
        } catch (Exception e) {
            started = false;
            throw new EngineException("Failed to loaded model.", e);
        } finally {
            if (!started) {
                stopPythonProcess(true);
            }
        }
    }

    synchronized void stopPythonProcess(boolean error) {
        restartCount.getAndIncrement();
        logger.info("Stop process: {}:{}, failure={}", workerId, pid, error);
        if (error) {
            int failures = model.intProperty("failed", 0);
            model.setProperty("failed", String.valueOf(failures + 1));
            logger.info("Failure count: {}", failures);
        }

        if (restartFuture != null) {
            try {
                if (!restartFuture.isDone()) {
                    if (!restartFuture.cancel(true)) {
                        logger.warn("Failed to cancel restart python process task.");
                    } else {
                        logger.info("Python process restart is cancelled.");
                    }
                }
            } catch (CancellationException ignore) {
                // ignore
            }
            restartFuture = null;
        }
        for (Connection conn : connections) {
            conn.disconnect();
        }
        if (process != null) {
            started = false;
            process.destroyForcibly();
            process = null;
        }
    }

    void setStarted(boolean started, int id) {
        if (restartCount.get() == id) {
            this.started = started;
            if (started) {
                latch.countDown();
            } else {
                while (latch.getCount() > 0) {
                    latch.countDown();
                }
            }
        }
    }

    boolean isStopped() {
        return !started;
    }

    static final class ReaderThread extends Thread {

        private InputStream is;
        private boolean error;
        private PyProcess lifeCycle;
        private int processId;

        public ReaderThread(
                String name, InputStream is, boolean error, PyProcess lifeCycle, int processId) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
            this.processId = processId;
        }

        @Override
        @SuppressWarnings("PMD.UseTryWithResources")
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8)) {
                while (scanner.hasNext()) {
                    String result = scanner.nextLine();
                    if (result == null) {
                        logger.warn("Got EOF: {}", getName());
                        break;
                    }
                    if (result.contains("Python engine started.")) {
                        logger.info("{}: {}", getName(), result);
                        lifeCycle.setStarted(true, processId);
                        continue;
                    }
                    int metricLoc = result.indexOf("[METRICS]");
                    if (metricLoc != -1) {
                        MODEL_METRIC.info("{}", Metric.parse(result.substring(metricLoc + 9)));
                        continue;
                    }
                    if (result.contains("ModelServerTelemetry:")) {
                        continue;
                    }

                    if (error) {
                        logger.warn("{}: {}", getName(), result);
                    } else {
                        logger.info("{}: {}", getName(), result);
                    }
                }
            } catch (Exception e) {
                logger.error("Couldn't create scanner - {}", getName(), e);
            } finally {
                logger.info("ReaderThread({}) stopped - {}", processId, getName());
                lifeCycle.setStarted(false, processId);
                try {
                    is.close();
                } catch (IOException e) {
                    logger.warn("Failed to close stream for thread - " + getName(), e);
                }
            }
        }
    }
}
