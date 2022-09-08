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
package ai.djl.serving.util;

import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;

/** A utility class to detect number of nueron cores. */
public final class NeuronUtils {

    private static final Logger logger = LoggerFactory.getLogger(NeuronUtils.class);

    private static final String TOKEN_URL = "http://169.254.169.254/latest/api/token";
    private static final String EC2_METADATA =
            "http://169.254.169.254/latest/meta-data/instance-type";

    private NeuronUtils() {}

    /**
     * Gets whether Neuron runtime library is in the system.
     *
     * @return {@code true} if Neuron runtime library is in the system
     */
    public static boolean hasNeuron() {
        return getNeuronCores() > 0;
    }

    /**
     * Returns the number of NeuronCores available in the system.
     *
     * @return the number of NeuronCores available in the system
     */
    public static int getNeuronCores() {
        String metadata = readMetadata();
        if (metadata == null) {
            return 0;
        }
        switch (metadata) {
            case "inf1.xlarge":
            case "inf1.2xlarge":
                return 4;
            case "inf1.6xlarge":
                return 16;
            case "inf1.24xlarge":
                return 64;
            default:
                return 0;
        }
    }

    private static String getToken() {
        try {
            String header = "X-aws-ec2-metadata-token-ttl-seconds";
            HttpURLConnection conn = openConnection(new URL(TOKEN_URL), "PUT", header, "21600");
            int statusCode = conn.getResponseCode();
            if (statusCode == HttpURLConnection.HTTP_OK) {
                try (InputStream is = conn.getInputStream()) {
                    return Utils.toString(is);
                }
            } else {
                logger.debug("EC2 IMDSv2: {} {}", statusCode, conn.getResponseMessage());
            }
            return null;
        } catch (IOException e) {
            logger.debug("Failed retrieve IMDSv2 token", e);
            return null;
        }
    }

    private static String readMetadata() {
        try {
            String header = "X-aws-ec2-metadata-token";
            String token = getToken();
            HttpURLConnection conn = openConnection(new URL(EC2_METADATA), "GET", header, token);
            int statusCode = conn.getResponseCode();
            if (statusCode == HttpURLConnection.HTTP_OK) {
                try (InputStream is = conn.getInputStream()) {
                    return Utils.toString(is);
                }
            } else {
                logger.debug("EC2 metadata: {} {}", statusCode, conn.getResponseMessage());
            }
            return null;
        } catch (IOException e) {
            logger.debug("Failed read ec2 metadata", e);
            return null;
        }
    }

    private static HttpURLConnection openConnection(
            URL url, String method, String header, String value) throws IOException {
        HttpURLConnection conn = (HttpURLConnection) url.openConnection(Proxy.NO_PROXY);
        conn.setConnectTimeout(1000);
        conn.setReadTimeout(1000);
        conn.setRequestMethod(method);
        conn.setDoOutput(true);
        conn.addRequestProperty("Accept", "*/*");
        conn.addRequestProperty("User-Agent", "djl");
        if (value != null) {
            conn.addRequestProperty(header, value);
        }
        conn.setInstanceFollowRedirects(false);
        conn.connect();
        return conn;
    }
}
