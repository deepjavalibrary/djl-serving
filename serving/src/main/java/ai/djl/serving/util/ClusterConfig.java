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
package ai.djl.serving.util;

import ai.djl.util.Utils;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

/** A class that holds cluster configurations. */
public final class ClusterConfig {

    private static final ClusterConfig INSTANCE = new ClusterConfig();

    private int clusterSize;
    private CountDownLatch latch;
    private String error;

    private ClusterConfig() {
        clusterSize = Integer.parseInt(Utils.getenv("DJL_CLUSTER_SIZE", "1"));
        latch = new CountDownLatch(clusterSize);
    }

    /**
     * Returns the {@code ClusterConfig} singleton object.
     *
     * @return the {@code ClusterConfig} singleton object
     */
    public static ClusterConfig getInstance() {
        return INSTANCE;
    }

    /**
     * Returns the cluster size.
     *
     * @return the cluster size
     */
    public int getClusterSize() {
        return clusterSize;
    }

    /**
     * Returns the error status message.
     *
     * @return the error status message
     */
    public String getError() {
        return error;
    }

    /**
     * Sets the error status message.
     *
     * @param error the error status message
     */
    public void setError(String error) {
        this.error = error;
    }

    /** Decreases the number of waiting workers. */
    public void countDown() {
        latch.countDown();
    }

    /**
     * Causes current threads to wait until all workers are ready.
     *
     * @throws InterruptedException if current thread is interrupted
     */
    public void await() throws InterruptedException {
        // TODO: support per model timeout
        int timeout = Integer.parseInt(Utils.getenv("MODEL_LOADING_TIMEOUT", "240"));
        if (!latch.await(timeout, TimeUnit.SECONDS)) {
            error = "Worker nodes timed out";
        }
    }
}
