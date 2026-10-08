/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.ModelException;
import ai.djl.util.Utils;

import io.netty.channel.epoll.Epoll;
import io.netty.channel.kqueue.KQueue;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class ConnectionTest {

    private static final String SWITCH = "SERVING_ENGINE_METRICS";

    @BeforeClass
    public void setUp() {
        System.setProperty("ENGINE_CACHE_DIR", "build/cache");
        if (System.getenv(SWITCH) != null) {
            throw new SkipException(SWITCH + " is set in the environment.");
        }
    }

    @AfterClass
    public void tearDown() {
        System.clearProperty("ENGINE_CACHE_DIR");
    }

    @AfterMethod(alwaysRun = true)
    public void clearSwitch() {
        System.clearProperty(SWITCH);
    }

    @Test
    public void testStartCommandWithoutEngineMetrics() throws IOException, ModelException {
        try (Model model = newModel()) {
            PyEnv pyEnv = new PyEnv(false);
            String[] expected = currentCommand(pyEnv, model, 19101);
            Assert.assertEquals(
                    Connection.getPythonStartCmd(pyEnv, model, 0, 19101, null), expected);
            Assert.assertEquals(expected.length, 19);

            for (String value : new String[] {"", "false", "FALSE", "0", "no", "tru"}) {
                System.setProperty(SWITCH, value);
                Assert.assertEquals(
                        Connection.getPythonStartCmd(pyEnv, model, 0, 19101, null),
                        expected,
                        value);
            }
        }
    }

    @Test
    public void testStartCommandWithEngineMetrics() throws IOException, ModelException {
        try (Model model = newModel()) {
            PyEnv pyEnv = new PyEnv(false);
            String[] current = currentCommand(pyEnv, model, 19102);
            for (String value : new String[] {"true", "TRUE", "True"}) {
                System.setProperty(SWITCH, value);
                String[] actual = Connection.getPythonStartCmd(pyEnv, model, 0, 19102, null);
                if (hasUnixSockets()) {
                    // everything else stays as it is, the arguments are appended
                    Assert.assertEquals(actual.length, 21, value);
                    Assert.assertEquals(Arrays.copyOf(actual, 19), current, value);
                    Assert.assertEquals(actual[19], "--metrics-sock-name");
                    Assert.assertEquals(
                            actual[20],
                            System.getProperty("java.io.tmpdir") + "/djl_sock.19102.metrics");
                } else {
                    // no Unix sockets (Windows): no exporter
                    Assert.assertEquals(actual, current, value);
                }
            }
        }
    }

    @Test
    public void testStartCommandInMpiMode() throws IOException, ModelException {
        try (Model model = newModel()) {
            PyEnv pyEnv = new PyEnv(true);
            pyEnv.setTensorParallelDegree(1);
            String[] current = Connection.getPythonStartCmd(pyEnv, model, 0, 19103, null);
            Assert.assertEquals(current.length, 47);

            System.setProperty(SWITCH, "true");
            String[] actual = Connection.getPythonStartCmd(pyEnv, model, 0, 19103, null);
            Assert.assertEquals(actual, current);
            Assert.assertFalse(Arrays.asList(actual).contains("--metrics-sock-name"));
        }
    }

    @Test
    public void testStartCommandInMultiNodeMode() throws Exception {
        Field field = PyEnv.class.getDeclaredField("clusterSize");
        field.setAccessible(true);
        int clusterSize = field.getInt(null);
        try (Model model = newModel()) {
            PyEnv pyEnv = new PyEnv(true);
            pyEnv.setTensorParallelDegree(2);
            String[] hosts = {"host0", "host1"};
            field.setInt(null, 2);
            String[] current = Connection.getPythonStartCmd(pyEnv, model, 0, 19104, hosts);
            Assert.assertEquals(current.length, 51);

            System.setProperty(SWITCH, "true");
            String[] actual = Connection.getPythonStartCmd(pyEnv, model, 0, 19104, hosts);
            Assert.assertEquals(actual, current);
            Assert.assertFalse(Arrays.asList(actual).contains("--metrics-sock-name"));
        } finally {
            field.setInt(null, clusterSize);
        }
    }

    @Test
    public void testMetricsSocketPath() {
        if (!hasUnixSockets()) {
            throw new SkipException("Unix sockets are not available.");
        }
        String tmp = System.getProperty("java.io.tmpdir");
        Connection connection =
                new Connection(new PyEnv(false), 45104, -1, "127.0.0.1", null, null);
        Assert.assertEquals(connection.getPort(), 19000 + 45104);
        Assert.assertEquals(connection.getMetricsSocketPath(), tmp + "/djl_sock.64104.metrics");

        // MPI workers are not started with an exporter
        Connection mpi = new Connection(new PyEnv(true), 45105, 0, "127.0.0.1", null, null);
        Assert.assertNull(mpi.getMetricsSocketPath());
    }

    @Test
    public void testStartRemovesLeftoverMetricsSocket() throws IOException, ModelException {
        if (!hasUnixSockets()) {
            throw new SkipException("Unix sockets are not available.");
        }
        Path off = Paths.get(System.getProperty("java.io.tmpdir"), "djl_sock.64107.metrics");
        Path on = Paths.get(System.getProperty("java.io.tmpdir"), "djl_sock.64108.metrics");
        Utils.deleteQuietly(off);
        Utils.deleteQuietly(on);
        Files.createFile(off);
        Files.createFile(on);
        try (Model model = newModel()) {
            PyEnv pyEnv = new PyEnv(false);
            // switched off: nothing is touched
            Process process = Connection.startPython(pyEnv, model, 0, 64107, null);
            process.destroyForcibly();
            Assert.assertTrue(Files.isRegularFile(off));

            // switched on: the python process creates the socket, a leftover has to go
            System.setProperty(SWITCH, "true");
            process = Connection.startPython(pyEnv, model, 0, 64108, null);
            try {
                // a socket that the new python process may have created by now is no regular file
                Assert.assertFalse(Files.isRegularFile(on));
            } finally {
                process.destroyForcibly();
            }
        } finally {
            Utils.deleteQuietly(off);
            Utils.deleteQuietly(on);
        }
    }

    @Test
    public void testDisconnectRemovesMetricsSocket() throws IOException {
        if (!hasUnixSockets()) {
            throw new SkipException("Unix sockets are not available.");
        }
        Connection connection =
                new Connection(new PyEnv(false), 45106, -1, "127.0.0.1", null, null);
        // a killed python process leaves both files behind
        Path request = Paths.get(System.getProperty("java.io.tmpdir"), "djl_sock.64106");
        Path metrics = Paths.get(connection.getMetricsSocketPath());
        Utils.deleteQuietly(request);
        Utils.deleteQuietly(metrics);
        Files.createFile(request);
        Files.createFile(metrics);
        try {
            connection.disconnect();
            Assert.assertFalse(Files.exists(request));
            Assert.assertFalse(Files.exists(metrics));
            // nothing to remove the second time
            connection.disconnect();
        } finally {
            Utils.deleteQuietly(request);
            Utils.deleteQuietly(metrics);
        }
    }

    /** Builds the command line as it was before engine metrics, for a CPU model and no MPI. */
    private static String[] currentCommand(PyEnv pyEnv, Model model, int port) {
        boolean uds = hasUnixSockets();
        String sock = System.getProperty("java.io.tmpdir") + "/djl_sock." + port;
        return new String[] {
            pyEnv.getPythonExecutable(),
            PyEnv.getEngineCacheDir() + "/djl_python_engine.py",
            "--sock-type",
            uds ? "unix" : "tcp",
            uds ? "--sock-name" : "--port",
            uds ? sock : String.valueOf(port),
            "--model-dir",
            model.getModelPath().toAbsolutePath().toString(),
            "--entry-point",
            "",
            "--device-id",
            "-1",
            "--cluster-size",
            String.valueOf(PyEnv.getClusterSize()),
            "--recommended-entry-point",
            "",
            "--log-level",
            pyEnv.getPythonLogLevel(),
            "--no-async-mode"
        };
    }

    private static boolean hasUnixSockets() {
        return Epoll.isAvailable() || KQueue.isAvailable();
    }

    private static Model newModel() throws IOException, ModelException {
        Model model = Model.newInstance("probe", Device.cpu(), "Python");
        model.load(Paths.get("src/test/resources/echo"));
        return model;
    }
}
