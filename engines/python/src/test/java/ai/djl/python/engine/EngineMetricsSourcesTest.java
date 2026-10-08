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
import ai.djl.util.Utils;

import io.netty.channel.epoll.Epoll;
import io.netty.channel.kqueue.KQueue;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;

public class EngineMetricsSourcesTest {

    private static final String SWITCH = "SERVING_ENGINE_METRICS";
    private static final String PREFIX = "engine_metrics_socket.";

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
    public void testDeviceLabel() {
        Assert.assertEquals(EngineMetricsSources.deviceLabel(Device.cpu()), "cpu");
        Assert.assertEquals(EngineMetricsSources.deviceLabel(Device.gpu(0)), "gpu0");
        Assert.assertEquals(EngineMetricsSources.deviceLabel(Device.gpu(12)), "gpu12");
        Assert.assertEquals(EngineMetricsSources.deviceLabel(Device.of("nc", 2)), "nc2");
        Assert.assertEquals(EngineMetricsSources.deviceLabel(Device.of("nc", -1)), "nc");
    }

    @Test
    public void testLabelsOnOneDevice() {
        Device device = Device.gpu(9101);
        try (Model m1 = newModel(device);
                Model m2 = newModel(device);
                Model m3 = newModel(device)) {
            // the first live worker on a device gets the bare label, the next ones a number
            String l1 = EngineMetricsSources.publish(m1, "/tmp/a.metrics");
            String l2 = EngineMetricsSources.publish(m2, "/tmp/b.metrics");
            String l3 = EngineMetricsSources.publish(m3, "/tmp/c.metrics");
            Assert.assertEquals(l1, "gpu9101");
            Assert.assertEquals(l2, "gpu9101-1");
            Assert.assertEquals(l3, "gpu9101-2");

            // the lowest free number is used again
            EngineMetricsSources.withdraw(m2, l2);
            Assert.assertEquals(EngineMetricsSources.publish(m2, "/tmp/b.metrics"), "gpu9101-1");
            EngineMetricsSources.withdraw(m1, l1);
            EngineMetricsSources.withdraw(m2, "gpu9101-1");
            Assert.assertEquals(EngineMetricsSources.publish(m1, "/tmp/a.metrics"), "gpu9101");
            Assert.assertEquals(EngineMetricsSources.publish(m2, "/tmp/b.metrics"), "gpu9101-1");

            EngineMetricsSources.withdraw(m1, "gpu9101");
            EngineMetricsSources.withdraw(m2, "gpu9101-1");
            EngineMetricsSources.withdraw(m3, l3);
            // everything is free again
            Assert.assertEquals(EngineMetricsSources.publish(m3, "/tmp/c.metrics"), "gpu9101");
            EngineMetricsSources.withdraw(m3, "gpu9101");
        }
    }

    @Test
    public void testDevicesAreIndependent() {
        try (Model gpu0 = newModel(Device.gpu(9102));
                Model gpu1 = newModel(Device.gpu(9103));
                Model gpu1Again = newModel(Device.gpu(9103));
                Model nc = newModel(Device.of("nc", 9104))) {
            Assert.assertEquals(EngineMetricsSources.publish(gpu0, "/tmp/0"), "gpu9102");
            Assert.assertEquals(EngineMetricsSources.publish(gpu1, "/tmp/1"), "gpu9103");
            Assert.assertEquals(EngineMetricsSources.publish(nc, "/tmp/2"), "nc9104");
            Assert.assertEquals(EngineMetricsSources.publish(gpu1Again, "/tmp/3"), "gpu9103-1");
            EngineMetricsSources.withdraw(gpu0, "gpu9102");
            EngineMetricsSources.withdraw(gpu1, "gpu9103");
            EngineMetricsSources.withdraw(nc, "nc9104");
            EngineMetricsSources.withdraw(gpu1Again, "gpu9103-1");
        }
    }

    @Test
    public void testParallelPublishAndWithdraw() throws Exception {
        int threads = 8;
        int rounds = 300;
        Device device = Device.gpu(9105);
        Map<String, Boolean> held = new ConcurrentHashMap<>();
        List<Model> models = new ArrayList<>();
        ExecutorService pool = Executors.newFixedThreadPool(threads);
        try {
            CountDownLatch start = new CountDownLatch(1);
            List<Future<Integer>> futures = new ArrayList<>();
            for (int i = 0; i < threads; ++i) {
                Model model = newModel(device);
                models.add(model);
                futures.add(
                        pool.submit(
                                () -> {
                                    start.await();
                                    int duplicates = 0;
                                    for (int r = 0; r < rounds; ++r) {
                                        String label =
                                                EngineMetricsSources.publish(model, "/tmp/p");
                                        if (held.put(label, Boolean.TRUE) != null) {
                                            ++duplicates;
                                        }
                                        LockSupport.parkNanos(20_000);
                                        held.remove(label);
                                        EngineMetricsSources.withdraw(model, label);
                                    }
                                    return duplicates;
                                }));
            }
            start.countDown();
            for (Future<Integer> future : futures) {
                // no label was ever held by two workers at once
                Assert.assertEquals(future.get(60, TimeUnit.SECONDS).intValue(), 0);
            }
            Assert.assertTrue(held.isEmpty());
            for (Model model : models) {
                Assert.assertTrue(engineMetricsProperties(model).isEmpty());
            }
            // every label was released
            Model last = models.get(0);
            Assert.assertEquals(EngineMetricsSources.publish(last, "/tmp/p"), "gpu9105");
            EngineMetricsSources.withdraw(last, "gpu9105");
        } finally {
            pool.shutdownNow();
            for (Model model : models) {
                model.close();
            }
        }
    }

    @Test
    public void testPublishAndWithdraw() {
        try (Model model = newModel(Device.gpu(9106))) {
            String label = EngineMetricsSources.publish(model, "/tmp/x.metrics");
            Assert.assertEquals(label, "gpu9106");
            Assert.assertEquals(model.getProperty(PREFIX + "gpu9106"), "/tmp/x.metrics");
            Assert.assertEquals(engineMetricsProperties(model).size(), 1);

            EngineMetricsSources.withdraw(model, label);
            Assert.assertNull(model.getProperty(PREFIX + "gpu9106"));
            Assert.assertTrue(engineMetricsProperties(model).isEmpty());

            // withdrawing a second time, or a label that is not published, changes nothing
            EngineMetricsSources.withdraw(model, label);
            EngineMetricsSources.withdraw(model, "gpu9106-3");
            EngineMetricsSources.withdraw(model, null);
            Assert.assertTrue(engineMetricsProperties(model).isEmpty());
        }
    }

    @Test
    public void testSecondWithdrawKeepsLabelOfNewHolder() {
        Device device = Device.gpu(9107);
        try (Model m1 = newModel(device);
                Model m2 = newModel(device);
                Model m3 = newModel(device)) {
            String first = EngineMetricsSources.publish(m1, "/tmp/1.metrics");
            EngineMetricsSources.withdraw(m1, first);
            // the label is handed to a new worker on another model
            String second = EngineMetricsSources.publish(m2, "/tmp/2.metrics");
            Assert.assertEquals(second, first);
            // a stale second withdraw of the first worker must not free it again
            EngineMetricsSources.withdraw(m1, first);
            String third = EngineMetricsSources.publish(m3, "/tmp/3.metrics");
            Assert.assertEquals(third, "gpu9107-1");
            Assert.assertEquals(m2.getProperty(PREFIX + "gpu9107"), "/tmp/2.metrics");

            EngineMetricsSources.withdraw(m2, second);
            EngineMetricsSources.withdraw(m3, third);
        }
    }

    @Test
    public void testRegisterEngineMetrics() throws Exception {
        if (!hasUnixSockets()) {
            throw new SkipException("Unix sockets are not available.");
        }
        Device device = Device.gpu(9108);
        try (Model model = newModel(device)) {
            PyProcess process = new PyProcess(model, new PyEnv(false), -1);
            Path socket = Paths.get(metricsSocketPath(process));
            Utils.deleteQuietly(socket);
            try {
                // switched off: nothing is published, even if the socket exists
                Files.createFile(socket);
                process.registerEngineMetrics();
                Assert.assertTrue(engineMetricsProperties(model).isEmpty());
                Files.delete(socket);

                // switched on, but the python process did not create the socket
                // (for example because prometheus_client is not installed)
                System.setProperty(SWITCH, "true");
                process.registerEngineMetrics();
                Assert.assertTrue(engineMetricsProperties(model).isEmpty());

                // switched on and the socket exists
                Files.createFile(socket);
                process.registerEngineMetrics();
                Map<String, String> sources = engineMetricsProperties(model);
                Assert.assertEquals(sources.size(), 1);
                Assert.assertEquals(sources.get(PREFIX + "gpu9108"), socket.toString());

                // registering again does not take a second label
                process.registerEngineMetrics();
                Assert.assertEquals(engineMetricsProperties(model).size(), 1);

                // stopping the python process (restart) keeps the registration
                process.stopPythonProcess(false);
                Assert.assertEquals(engineMetricsProperties(model).size(), 1);
                Assert.assertFalse(Files.exists(socket));

                // cleaning up removes it, and frees the label
                Files.createFile(socket);
                process.cleanUp();
                Assert.assertTrue(engineMetricsProperties(model).isEmpty());
                Assert.assertFalse(Files.exists(socket));
                Assert.assertEquals(EngineMetricsSources.publish(model, "/tmp/z"), "gpu9108");
                EngineMetricsSources.withdraw(model, "gpu9108");

                // cleaning up twice is safe
                process.cleanUp();
                Assert.assertEquals(EngineMetricsSources.publish(model, "/tmp/z"), "gpu9108");
                EngineMetricsSources.withdraw(model, "gpu9108");
            } finally {
                Utils.deleteQuietly(socket);
            }
        }
    }

    @Test
    public void testRegisterEngineMetricsInMpiMode() throws Exception {
        if (!hasUnixSockets()) {
            throw new SkipException("Unix sockets are not available.");
        }
        try (Model model = newModel(Device.gpu(9109))) {
            PyEnv mpi = new PyEnv(true);
            mpi.setTensorParallelDegree(1);
            PyProcess process = new PyProcess(model, mpi, 0);
            System.setProperty(SWITCH, "true");
            // MPI workers are not started with an exporter, so there is no socket to publish
            Assert.assertNull(connection(process).getMetricsSocketPath());
            process.registerEngineMetrics();
            Assert.assertTrue(engineMetricsProperties(model).isEmpty());
        }
    }

    private static Model newModel(Device device) {
        return Model.newInstance("probe", device, "Python");
    }

    private static Map<String, String> engineMetricsProperties(Model model) {
        Map<String, String> ret = new ConcurrentHashMap<>();
        for (Map.Entry<String, String> entry : model.getProperties().entrySet()) {
            if (entry.getKey().startsWith(PREFIX)) {
                ret.put(entry.getKey(), entry.getValue());
            }
        }
        return ret;
    }

    private static boolean hasUnixSockets() {
        return Epoll.isAvailable() || KQueue.isAvailable();
    }

    private static Connection connection(PyProcess process) throws ReflectiveOperationException {
        Field field = PyProcess.class.getDeclaredField("connections");
        field.setAccessible(true);
        List<?> connections = (List<?>) field.get(process);
        return (Connection) connections.get(0);
    }

    private static String metricsSocketPath(PyProcess process) throws ReflectiveOperationException {
        return connection(process).getMetricsSocketPath();
    }
}
