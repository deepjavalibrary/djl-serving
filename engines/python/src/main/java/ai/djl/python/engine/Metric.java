/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import com.google.gson.annotations.SerializedName;
import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** A class represents a metric object. */
public class Metric {

    private static final Pattern PATTERN =
            Pattern.compile(
                    "\\s*([\\w\\s]+)\\.([\\w\\s]+):([0-9\\-,.e]+)\\|#([^|]*)\\|#hostname:([^,]+),([^,]+)(,(.*))?");

    @SerializedName("MetricName")
    private String metricName;

    @SerializedName("Value")
    private String value;

    @SerializedName("Unit")
    private String unit;

    @SerializedName("Dimensions")
    private List<Dimension> dimensions;

    @SerializedName("Timestamp")
    private String timestamp;

    @SerializedName("RequestId")
    private String requestId;

    @SerializedName("HostName")
    private String hostName;

    /** Constructs a new {@code Metric} instance. */
    public Metric() {}

    /**
     * Constructs a new {@code Metric} instance.
     *
     * @param metricName the metric name
     * @param unit the metric unit
     * @param value the metric value
     * @param hostName the host name
     * @param timestamp the metric timestamp
     * @param requestId the request ID
     */
    public Metric(
            String metricName,
            String unit,
            String value,
            String hostName,
            String timestamp,
            String requestId) {
        this.metricName = metricName;
        this.value = value;
        this.unit = unit;
        this.hostName = hostName;
        this.timestamp = timestamp;
        this.requestId = requestId;
    }

    /**
     * Returns the host name.
     *
     * @return the host name
     */
    public String getHostName() {
        return hostName;
    }

    /**
     * Returns the request id.
     *
     * @return the request id
     */
    public String getRequestId() {
        return requestId;
    }

    /**
     * Sets the request id.
     *
     * @param requestId the request id
     */
    public void setRequestId(String requestId) {
        this.requestId = requestId;
    }

    /**
     * Returns the metric name.
     *
     * @return the metric name
     */
    public String getMetricName() {
        return metricName;
    }

    /**
     * Returns the metric value.
     *
     * @return the metric value
     */
    public String getValue() {
        return value;
    }

    /**
     * Returns the metric unit.
     *
     * @return the metric unit
     */
    public String getUnit() {
        return unit;
    }

    /**
     * Returns the metric dimensions.
     *
     * @return the metric dimensions
     */
    public List<Dimension> getDimensions() {
        return dimensions;
    }

    /**
     * Sets the metric dimensions.
     *
     * @param dimensions the metric dimensions
     */
    public void setDimensions(List<Dimension> dimensions) {
        this.dimensions = dimensions;
    }

    /**
     * Returns the metric timestamp.
     *
     * @return the metric timestamp
     */
    public String getTimestamp() {
        return timestamp;
    }

    /**
     * Returns a {@code Metric} instance parsed from the log string.
     *
     * @param line the input string
     * @return a {@code Metric} object
     */
    public static Metric parse(String line) {
        // DiskAvailable.Gigabytes:311|#Level:Host|#hostname:localhost,1650953744320,request_id
        Matcher matcher = PATTERN.matcher(line);
        if (!matcher.matches()) {
            return null;
        }

        Metric metric =
                new Metric(
                        matcher.group(1),
                        matcher.group(2),
                        matcher.group(3),
                        matcher.group(5),
                        matcher.group(6),
                        matcher.group(8));

        String dimensions = matcher.group(4);
        if (dimensions != null) {
            String[] dimension = dimensions.split(",");
            List<Dimension> list = new ArrayList<>(dimension.length);
            for (String dime : dimension) {
                String[] pair = dime.split(":");
                if (pair.length == 2) {
                    list.add(new Dimension(pair[0], pair[1]));
                }
            }
            metric.setDimensions(list);
        }

        return metric;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(128);
        sb.append(metricName).append('.').append(unit).append(':').append(getValue()).append("|#");
        if (dimensions != null) {
            boolean first = true;
            for (Dimension dimension : dimensions) {
                if (first) {
                    first = false;
                } else {
                    sb.append(',');
                }
                sb.append(dimension.getName()).append(':').append(dimension.getValue());
            }
        }
        sb.append("|#hostname:").append(hostName);
        if (requestId != null) {
            sb.append(",requestID:").append(requestId);
        }
        sb.append(",timestamp:").append(timestamp);
        return sb.toString();
    }
}
