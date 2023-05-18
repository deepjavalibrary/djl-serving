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
import ai.djl.util.Platform;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import io.netty.channel.EventLoopGroup;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Python engine environment. */
public class PyEnv {

    static final Logger logger = LoggerFactory.getLogger(PyEnv.class);

    private static String engineCacheDir;
    private static String version;
    private static EventLoopGroup eventLoopGroup;

    private boolean mpiMode;
    private String pythonExecutable;
    private String entryPoint;
    private String handler;
    private int predictTimeout;
    private int modelLoadingTimeout;
    private int tensorParallelDegree;
    private Map<String, String> envs;
    private Map<String, String> initParameters;
    private boolean initialized;

    private boolean failOnInitialize = true;

    /**
     * Constructs a new {@code PyEnv} instance.
     *
     * @param mpiMode true to use MPI launcher
     */
    public PyEnv(boolean mpiMode) {
        this.mpiMode = mpiMode;
        pythonExecutable = Utils.getenv("PYTHON_EXECUTABLE");
        if (pythonExecutable == null) {
            pythonExecutable = "python3";
        }
        handler = "handle";
        envs = new ConcurrentHashMap<>();
        initParameters = new ConcurrentHashMap<>();
    }

    static synchronized void init() {
        if (eventLoopGroup != null) {
            return;
        }

        eventLoopGroup = Connection.newEventLoopGroup();

        Path tmp = null;
        try {
            Platform platform = Platform.detectPlatform("python");
            version = platform.getVersion();
            Path cacheDir = Utils.getEngineCacheDir("python");
            logger.debug("Using cache dir: {}", cacheDir);
            Path path = cacheDir.resolve(version);
            engineCacheDir = path.toAbsolutePath().toString();
            if (Files.exists(path)) {
                return;
            }

            Files.createDirectories(cacheDir);
            tmp = Files.createTempDirectory(cacheDir, "tmp");
            Files.createDirectories(tmp.resolve("djl_python"));
            for (String file : platform.getLibraries()) {
                String libPath = '/' + file;
                logger.info("Extracting {} to cache ...", libPath);
                try (InputStream is = PyEnv.class.getResourceAsStream(libPath)) {
                    if (is == null) {
                        throw new AssertionError("Python engine script not found: " + libPath);
                    }
                    Files.copy(is, tmp.resolve(file), StandardCopyOption.REPLACE_EXISTING);
                }
            }
            Utils.moveQuietly(tmp, path);
            tmp = null;
        } catch (IOException e) {
            throw new EngineException("Failed to initialize python engine.", e);
        } finally {
            if (tmp != null) {
                Utils.deleteQuietly(tmp);
            }
        }
    }

    static String getVersion() {
        return version;
    }

    static String getEngineCacheDir() {
        return engineCacheDir;
    }

    static EventLoopGroup getEventLoopGroup() {
        return eventLoopGroup;
    }

    /**
     * Adds an environment variable.
     *
     * @param key the environment variable name
     * @param value the environment variable value
     */
    public void addEnv(String key, String value) {
        envs.put(key, value);
    }

    /**
     * Adds a model initialization parameter.
     *
     * @param key the environment variable name
     * @param value the environment variable value
     */
    public void addParameter(String key, String value) {
        initParameters.put(key, value);
    }

    /**
     * Returns the python model initialization parameters.
     *
     * @return the python model initialization parameters
     */
    public Map<String, String> getInitParameters() {
        return initParameters;
    }

    /**
     * Installs model dependencies if needed.
     *
     * @param modelDir the model directory
     */
    public synchronized void installDependency(Path modelDir) {
        if (initialized) {
            return;
        }
        Path file = modelDir.resolve("requirements.txt");
        if (Files.isRegularFile(file)) {
            List<String> cmd = new ArrayList<>(9);
            cmd.add(pythonExecutable);
            cmd.add("-m");
            cmd.add("pip");
            if (!logger.isDebugEnabled()) {
                cmd.add("-q");
            }
            cmd.add("install");
            cmd.add("-r");
            cmd.add(file.toAbsolutePath().toString());
            if (Boolean.getBoolean("offline")) {
                cmd.add("--no-deps");
            }
            Path dir = modelDir.resolve("requirements");
            if (Files.isDirectory(dir)) {
                cmd.add("-f");
                cmd.add(file.toAbsolutePath().toString());
            }
            try {
                logger.info("Found requirements.txt, start installing Python dependencies...");
                logger.debug("{}", cmd);
                Process process = new ProcessBuilder(cmd).redirectErrorStream(true).start();
                String logOutput;
                try (InputStream is = process.getInputStream()) {
                    logOutput = Utils.toString(is);
                }
                int ret = process.waitFor();
                if (ret == 0) {
                    logger.info("pip install requirements succeed!");
                    logger.debug("{}", logOutput);
                } else {
                    logger.warn("pip install failed with error code: {}", ret);
                    logger.warn("{}", logOutput);
                }
            } catch (IOException | InterruptedException e) {
                logger.warn("pip install requirements failed.", e);
            }
        }
        initialized = true;
    }

    boolean isMpiMode() {
        return mpiMode;
    }

    /**
     * Returns the python executable path.
     *
     * @return the python executable path
     */
    public String getPythonExecutable() {
        return pythonExecutable;
    }

    /**
     * Sets the python executable path.
     *
     * @param pythonExecutable the python executable path
     */
    public void setPythonExecutable(String pythonExecutable) {
        this.pythonExecutable = pythonExecutable;
    }

    /**
     * Returns the tensor parallel degree.
     *
     * @return the tensor parallel degree
     */
    public int getTensorParallelDegree() {
        if (tensorParallelDegree == 0) {
            String value = Utils.getenv("TENSOR_PARALLEL_DEGREE");
            if (value != null) {
                tensorParallelDegree = Integer.parseInt(value);
            }
        }
        return tensorParallelDegree;
    }

    /**
     * Sets the tensor parallel degree.
     *
     * @param tensorParallelDegree the tensor parallel degree
     */
    public void setTensorParallelDegree(int tensorParallelDegree) {
        this.tensorParallelDegree = tensorParallelDegree;
    }

    int getMpiWorkers() {
        int gpuCount = CudaUtils.getGpuCount();
        return gpuCount / getTensorParallelDegree();
    }

    /**
     * Returns the model's entrypoint file path.
     *
     * @return the model's entrypoint file path
     */
    public String getEntryPoint() {
        return entryPoint == null ? "model.py" : entryPoint;
    }

    /**
     * Sets the model's entrypoint file path.
     *
     * @param entryPoint the model's entrypoint file path
     */
    public void setEntryPoint(String entryPoint) {
        this.entryPoint = entryPoint;
    }

    /**
     * Returns the python model's handler function.
     *
     * @return the python file's handler function
     */
    public String getHandler() {
        return handler;
    }

    /**
     * Sets the python model's handler function.
     *
     * @param handler the python file's handler function
     */
    public void setHandler(String handler) {
        this.handler = handler;
    }

    /**
     * Returns the prediction timeout in seconds.
     *
     * @return the prediction timeout in seconds
     */
    public int getPredictTimeout() {
        if (predictTimeout == 0) {
            predictTimeout = getDefaultTimeout("PREDICT_TIMEOUT", 120);
        }
        return predictTimeout;
    }

    /**
     * Sets the prediction timeout in seconds.
     *
     * @param predictTimeout the prediction timeout in seconds
     */
    public void setPredictTimeout(int predictTimeout) {
        this.predictTimeout = predictTimeout;
    }

    /**
     * Returns the model loading timeout in seconds.
     *
     * @return the model loading timeout in seconds
     */
    public int getModelLoadingTimeout() {
        if (modelLoadingTimeout == 0) {
            modelLoadingTimeout = getDefaultTimeout("MODEL_LOADING_TIMEOUT", 240);
        }
        return modelLoadingTimeout;
    }

    /**
     * Returns true to forcibly fail if initialize process in python failed.
     *
     * @return true if forcibly failed
     */
    public boolean isFailOnInitialize() {
        return failOnInitialize;
    }

    /**
     * Enables to forcibly fail if initialize process in python failed.
     *
     * @param failOnInitialize the flag
     */
    public void setFailOnInitialize(boolean failOnInitialize) {
        this.failOnInitialize = failOnInitialize;
    }

    /**
     * Sets the model loading timeout in seconds.
     *
     * @param modelLoadingTimeout the model loading timeout in seconds
     */
    public void setModelLoadingTimeout(int modelLoadingTimeout) {
        this.modelLoadingTimeout = modelLoadingTimeout;
    }

    String[] getEnvironmentVars(Model model) {
        ArrayList<String> envList = new ArrayList<>();
        StringBuilder pythonPath = new StringBuilder();
        HashMap<String, String> environment = new HashMap<>(Utils.getenv());
        if (Utils.getenv("PYTHONPATH") != null) {
            pythonPath.append(Utils.getenv("PYTHONPATH")).append(File.pathSeparatorChar);
        }
        pythonPath.append(engineCacheDir).append(File.pathSeparatorChar);
        pythonPath.append(model.getModelPath().toAbsolutePath());

        environment.put("PYTHONPATH", pythonPath.toString());
        environment.putAll(envs);
        for (Map.Entry<String, String> entry : environment.entrySet()) {
            envList.add(entry.getKey() + '=' + entry.getValue());
        }

        return envList.toArray(new String[0]);
    }

    private static int getDefaultTimeout(String key, int def) {
        String timeout = Utils.getenv(key);
        if (timeout == null) {
            return def;
        }
        try {
            return Integer.parseInt(timeout);
        } catch (NumberFormatException e) {
            logger.warn("Invalid timeout value: {}.", timeout);
        }
        return def;
    }
}
