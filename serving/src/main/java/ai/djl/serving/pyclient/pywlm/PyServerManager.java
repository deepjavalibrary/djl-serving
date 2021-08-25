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
package ai.djl.serving.pyclient.pywlm;

import ai.djl.serving.pyclient.PythonConnector;
import ai.djl.serving.util.ConfigManager;

/** This class manages the python server connections. */
public final class PyServerManager {
    private static final String DEFAULT_SOCKET_PATH = "/tmp/uds_sock";

    private ConfigManager configManager;
    private static PyServerManager pyServerManager;
    private PyWorkLoadManager wlm;
    private int noOfPythonWorkers;

    private PyServerManager(ConfigManager configManager) {
        this.configManager = configManager;
        boolean startPythonServer =
                Boolean.getBoolean(configManager.getProperty("startPythonServer", "false"));
        if (!startPythonServer) {
            return;
        }

        noOfPythonWorkers = Integer.parseInt(configManager.getProperty("noOfPythonWorkers", "0"));

        if (noOfPythonWorkers > 0) {
            this.wlm = new PyWorkLoadManager(noOfPythonWorkers);
        }
    }

    /**
     * Initializes the {@code PyServerManager} instance.
     *
     * @param configManager configmanager
     */
    public static void init(ConfigManager configManager) {
        pyServerManager = new PyServerManager(configManager);
    }

    /**
     * Returns the {@code PyServerManager} instance.
     *
     * @return PyServerManager
     */
    public static PyServerManager getInstance() {
        return pyServerManager;
    }

    /** Starts the python servers. */
    public void startServers() {
        boolean uds = configManager.useNativeIo();
        String pythonPath = configManager.getProperty("pythonPath", "python");
        int port = 9000;
        for (int i = 0; i < noOfPythonWorkers; i++) {
            port += i;
            PythonConnector connector;
            if (uds) {
                connector = new PythonConnector(true, null, -1, DEFAULT_SOCKET_PATH);
            } else {
                connector = new PythonConnector(false, "127.0.0.1", port, "null");
            }
            wlm.addThread(connector, pythonPath);
        }
    }

    /**
     * Adds a python job to the job queue.
     *
     * @param job python job
     * @return whether job is added or not
     */
    public boolean addJob(PyJob job) {
        return wlm.addJob(job);
    }
}
