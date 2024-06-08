/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.preometheus;

import ai.djl.metric.Metric;
import ai.djl.metric.MetricType;
import ai.djl.metric.Unit;
import ai.djl.prometheus.MetricExporter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.HashSet;

public class MetricExporterTest {

    @Test
    public void testExport() throws IOException {
        Logger metrics = LoggerFactory.getLogger("test_metric");
        metrics.info("{}", new Metric("MyCounter", 1));
        metrics.info("{}", new Metric("MyGauge", 1, Unit.MICROSECONDS));
        metrics.info("{}", new Metric("MyHistogram", MetricType.HISTOGRAM, 1, Unit.MICROSECONDS));
        try (ByteArrayOutputStream os = new ByteArrayOutputStream()) {
            MetricExporter.export(os, new HashSet<>());
            String res = os.toString(StandardCharsets.UTF_8);
            System.out.println(res);
            Assert.assertTrue(res.contains("MyCounter"));
            Assert.assertTrue(res.contains("MyGauge"));
            Assert.assertTrue(res.contains("MyHistogram"));
        }
    }

    @Test
    public void testPrometheusDisabled() throws IOException {
        System.setProperty("SERVING_PROMETHEUS", "false");
        try (ByteArrayOutputStream os = new ByteArrayOutputStream()) {
            Assert.assertThrows(() -> MetricExporter.export(os, new HashSet<>()));
        } finally {
            System.setProperty("SERVING_PROMETHEUS", "true");
        }
    }
}
