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
package ai.djl.prometheus;

import ai.djl.metric.Dimension;
import ai.djl.metric.Metric;
import ai.djl.metric.Unit;
import ai.djl.util.Utils;

import io.prometheus.metrics.core.metrics.Counter;
import io.prometheus.metrics.core.metrics.Gauge;
import io.prometheus.metrics.model.snapshots.Labels;

import org.apache.logging.log4j.core.LogEvent;
import org.apache.logging.log4j.core.appender.AbstractAppender;
import org.apache.logging.log4j.core.config.Property;
import org.apache.logging.log4j.core.config.plugins.Plugin;
import org.apache.logging.log4j.core.config.plugins.PluginAttribute;
import org.apache.logging.log4j.core.config.plugins.PluginFactory;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A log4j2 appender class to collect prometheus metrics. */
@Plugin(name = "Prometheus", category = "Core", elementType = "appender")
public final class PrometheusAppender extends AbstractAppender {

    private static final Map<String, Counter> COUNTERS = new ConcurrentHashMap<>();
    private static final Map<String, Gauge> GAUGES = new ConcurrentHashMap<>();

    private boolean usePrometheus;

    private PrometheusAppender(String name) {
        super(name, null, null, true, Property.EMPTY_ARRAY);
        usePrometheus = Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_PROMETHEUS"));
    }

    /** {@inheritDoc} */
    @Override
    public void append(LogEvent event) {
        if (usePrometheus) {
            Metric metric = (Metric) event.getMessage().getParameters()[0];
            String name = metric.getMetricName();
            Unit unit = metric.getUnit();
            Dimension[] dimension = metric.getDimensions();
            String labelName = dimension[0].getName();
            String labelValue = dimension[0].getValue();
            Labels labels = Labels.of(labelName, labelValue);
            if (unit == Unit.COUNT) {
                Counter counter =
                        COUNTERS.computeIfAbsent(name, k -> newCounter(name, labelName, unit));
                counter.labelValues(labelValue).incWithExemplar(metric.getValue(), labels);
            } else {
                Gauge gauge = GAUGES.computeIfAbsent(name, k -> newGauge(name, labelName, unit));
                gauge.labelValues(labelValue).setWithExemplar(metric.getValue(), labels);
            }
        }
    }

    private Counter newCounter(String name, String labelName, Unit unit) {
        return Counter.builder()
                .name(name)
                .labelNames(labelName)
                .help(": prometheus counter metric, unit: " + unit.getValue())
                .register();
    }

    private Gauge newGauge(String name, String labelName, Unit unit) {
        return Gauge.builder()
                .name(name)
                .labelNames(labelName)
                .help(": prometheus gauge metric, unit: " + unit.getValue())
                .register();
    }

    /**
     * Constructs a new {@code PrometheusAppender} instance.
     *
     * @param name the appender name
     * @return a new {@code PrometheusAppender} instance
     */
    @PluginFactory
    public static PrometheusAppender createAppender(@PluginAttribute("name") String name) {
        return new PrometheusAppender(name);
    }
}
