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

import ai.djl.util.Utils;

import io.prometheus.metrics.expositionformats.PrometheusTextFormatWriter;
import io.prometheus.metrics.model.registry.PrometheusRegistry;

import java.io.IOException;
import java.io.OutputStream;
import java.util.Set;

/** A utility class to export prometheus metrics. */
public final class MetricExporter {

    public static final String CONTENT_TYPE = PrometheusTextFormatWriter.CONTENT_TYPE;
    private static final PrometheusTextFormatWriter WRITER = new PrometheusTextFormatWriter(false);

    private MetricExporter() {}

    /**
     * Writes prometheus metrics to {@code OutputStream}.
     *
     * @param os the {@code OutputStream} to write
     * @param set the filter names
     * @throws IOException throws if failed to write
     */
    public static void export(OutputStream os, Set<String> set) throws IOException {
        if (!Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_PROMETHEUS"))) {
            throw new IllegalArgumentException(
                    "Prometheus is not enabled, set SERVING_PROMETHEUS environment var to true to"
                            + " enable prometheus metrics");
        }
        WRITER.write(
                os,
                PrometheusRegistry.defaultRegistry.scrape(s -> set.isEmpty() || set.contains(s)));
    }
}
