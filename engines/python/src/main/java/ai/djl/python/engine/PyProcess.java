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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.EngineException;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Scanner;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

class PyProcess {

    static final Logger logger = LoggerFactory.getLogger(PyProcess.class);

    private PyEnv pyEnv;
    private Model model;
    private Process process;
    private Connection connection;
    private CountDownLatch latch;
    private boolean started;
    private ReaderThread err;
    private ReaderThread out;
    private CompletableFuture<Void> restartFuture;

    PyProcess(Model model, PyEnv pyEnv) {
        this.model = model;
        this.pyEnv = pyEnv;
        connection = new Connection();
        latch = new CountDownLatch(1);
    }

    Output predict(Input inputs, int timeout, boolean restart) throws TranslateException {
        try {
            if (inputs.getProperty("handler", null) == null) {
                String handler = pyEnv.getHandler();
                if (handler != null) {
                    inputs.addProperty("handler", handler);
                }
            }
            Device device = model.getNDManager().getDevice();
            inputs.addProperty("device_id", String.valueOf(device.getDeviceId()));
            CompletableFuture<Output> future = connection.send(inputs);
            return future.get(timeout, TimeUnit.SECONDS);
        } catch (Exception e) {
            if (restart) {
                stopPythonProcess();
                logger.info("Restart python process ...");
                restartFuture = CompletableFuture.runAsync(this::startPythonProcess);
            }
            throw new TranslateException(e);
        }
    }

    synchronized void startPythonProcess() {
        try {
            pyEnv.installDependency(model.getModelPath());
            process = connection.startPython(pyEnv, model);

            String modelName = model.getName();
            modelName = modelName.substring(0, Math.min(modelName.length(), 15));
            String threadName = "W-" + connection.getPort() + '-' + modelName;
            err = new ReaderThread(threadName, process.getErrorStream(), true, this);
            out = new ReaderThread(threadName, process.getInputStream(), false, this);
            err.start();
            out.start();
            if (!latch.await(2, TimeUnit.MINUTES)) {
                throw new EngineException("Python process startup time out.");
            }
            if (!started) {
                throw new EngineException(
                        "Python stream closed unexpectedly, exit code: " + process.exitValue());
            }

            connection.connect();

            try {
                // initialize model with an empty request
                predict(new Input(), pyEnv.getModelLoadingTimeout(), false);
            } catch (TranslateException e) {
                throw new EngineException("Failed to load Python model.", e);
            }
        } catch (InterruptedException e) {
            throw new EngineException("Worker startup cancelled.", e);
        } catch (IOException e) {
            throw new EngineException("Failed connect to Python worker process.", e);
        } finally {
            if (!started) {
                stopPythonProcess();
            }
        }
    }

    synchronized void stopPythonProcess() {
        if (restartFuture != null) {
            try {
                if (!restartFuture.isDone()) {
                    if (!restartFuture.cancel(true)) {
                        logger.warn("Failed to cancel restart python process task.");
                    }
                }
            } catch (CancellationException ignore) {
                // ignore
            }
            restartFuture = null;
        }
        if (process != null) {
            started = false;
            if (err != null) {
                err.shutdown();
            }
            if (out != null) {
                out.shutdown();
            }
            connection.disconnect();
            process.destroyForcibly();
            process = null;
        }
    }

    void setStarted(boolean started) {
        this.started = started;
        latch.countDown();
    }

    boolean isStopped() {
        return !started;
    }

    private static final class ReaderThread extends Thread {

        private InputStream is;
        private boolean error;
        private PyProcess lifeCycle;
        private AtomicBoolean isRunning = new AtomicBoolean(true);

        public ReaderThread(String name, InputStream is, boolean error, PyProcess lifeCycle) {
            super(name + (error ? "-stderr" : "-stdout"));
            this.is = is;
            this.error = error;
            this.lifeCycle = lifeCycle;
        }

        public void shutdown() {
            isRunning.set(false);
        }

        @Override
        @SuppressWarnings("PMD.UseTryWithResources")
        public void run() {
            try (Scanner scanner = new Scanner(is, StandardCharsets.UTF_8.name())) {
                while (isRunning.get() && scanner.hasNext()) {
                    String result = scanner.nextLine();
                    if (result == null) {
                        break;
                    }
                    if ("Python engine started.".equals(result)) {
                        lifeCycle.setStarted(true);
                    }
                    if (error) {
                        logger.warn(result);
                    } else {
                        logger.info(result);
                    }
                }
            } catch (Exception e) {
                logger.error("Couldn't create scanner - {}", getName(), e);
            } finally {
                logger.info("Stopped Scanner - {}", getName());
                lifeCycle.setStarted(false);
                try {
                    is.close();
                } catch (IOException e) {
                    logger.error("Failed to close stream for thread {}", this.getName(), e);
                }
            }
        }
    }
}
