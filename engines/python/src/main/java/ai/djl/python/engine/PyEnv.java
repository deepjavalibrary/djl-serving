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
import io.netty.channel.EventLoopGroup;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Python engine environment. */
public class PyEnv {

    static final Logger logger = LoggerFactory.getLogger(PyEnv.class);

    private static String engineCacheDir;
    private static String version;
    private static EventLoopGroup eventLoopGroup;

    private String pythonExecutable;
    private String entryPoint;
    private String handler;
    private Map<String, String> envs;

    /** Constructs a new {@code PyEnv} instance. */
    public PyEnv() {
        pythonExecutable = System.getenv("PYTHON_EXECUTABLE");
        if (pythonExecutable == null) {
            pythonExecutable = "python3";
        }
        handler = "handle";
        envs = new ConcurrentHashMap<>();
    }

    static void init() {
        eventLoopGroup = Connection.newEventLoopGroup();

        Path tmp = null;
        try {
            URL url = PyEnv.class.getResource("/ai/djl/python/python.properties");
            if (url == null) {
                throw new AssertionError("python.properties is missing in jar.");
            }
            Platform platform = Platform.fromUrl(url);
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
     * Add an environment variable.
     *
     * @param key the environment variable name
     * @param value the environment variable value
     */
    public void addEnv(String key, String value) {
        envs.put(key, value);
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

    String[] getEnvironmentVars(Model model) {
        ArrayList<String> envList = new ArrayList<>();
        StringBuilder pythonPath = new StringBuilder();
        HashMap<String, String> environment = new HashMap<>(System.getenv());
        if (System.getenv("PYTHONPATH") != null) {
            pythonPath.append(System.getenv("PYTHONPATH")).append(File.pathSeparatorChar);
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
}
