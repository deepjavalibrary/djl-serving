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
package ai.djl.serving.wlm.util;

/** This manages some configurations used by the {@link ai.djl.serving.wlm.WorkLoadManager}. */
public final class WlmConfigManager {

    private int jobQueueSize = 1000;
    private int maxIdleSeconds = 60;
    private int batchSize = 1;
    private int maxBatchDelayMillis = 100;
    private int reservedMemoryMb = 500;
    private String loadOnDevices;

    private static final WlmConfigManager INSTANCE = new WlmConfigManager();

    private WlmConfigManager() {}

    /**
     * Returns the singleton {@code ConfigManager} instance.
     *
     * @return the singleton {@code ConfigManager} instance
     */
    public static WlmConfigManager getInstance() {
        return INSTANCE;
    }

    /**
     * Returns if debug is enabled.
     *
     * @return {@code true} if debug is enabled
     */
    public boolean isDebug() {
        return Boolean.getBoolean("ai.djl.serving.debug");
    }

    /**
     * Returns the default job queue size.
     *
     * @return the default job queue size
     */
    public int getJobQueueSize() {
        return jobQueueSize;
    }

    /**
     * Sets the default job queue size.
     *
     * @param jobQueueSize the new default job queue size
     */
    public void setJobQueueSize(int jobQueueSize) {
        this.jobQueueSize = jobQueueSize;
    }

    /**
     * Returns the default max idle time for workers.
     *
     * @return the default max idle time
     */
    public int getMaxIdleSeconds() {
        return maxIdleSeconds;
    }

    /**
     * Sets the default max idle time in seconds for workers.
     *
     * @param maxIdleSeconds the new default max idle time in seconds
     */
    public void setMaxIdleSeconds(int maxIdleSeconds) {
        this.maxIdleSeconds = maxIdleSeconds;
    }

    /**
     * Returns the default batchSize for workers.
     *
     * @return the default max idle time
     */
    public int getBatchSize() {
        return batchSize;
    }

    /**
     * Sets the default batchSize for workers.
     *
     * @param batchSize the new default batchSize
     */
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }

    /**
     * Returns the default max batch delay in milliseconds for the working queue.
     *
     * @return the default max batch delay in milliseconds
     */
    public int getMaxBatchDelayMillis() {
        return maxBatchDelayMillis;
    }

    /**
     * Sets the default max batch delay in milliseconds for the working queue.
     *
     * @param maxBatchDelayMillis the new default max batch delay in milliseconds
     */
    public void setMaxBatchDelayMillis(int maxBatchDelayMillis) {
        this.maxBatchDelayMillis = maxBatchDelayMillis;
    }

    /**
     * Returns the default reserved memory in MB.
     *
     * @return the default reserved memory in MB
     */
    public int getReservedMemoryMb() {
        return reservedMemoryMb;
    }

    /**
     * Sets the reserved memory in MB.
     *
     * @param reservedMemoryMb the reserved memory in MB
     */
    public void setReservedMemoryMb(int reservedMemoryMb) {
        this.reservedMemoryMb = reservedMemoryMb;
    }

    /**
     * Returns the devices the model will be loaded on at startup.
     *
     * @return the devices the model will be loaded on at startup
     */
    public String getLoadOnDevices() {
        return loadOnDevices;
    }

    /**
     * Sets the devices the model will be loaded on at startup.
     *
     * @param loadOnDevices thes the default model will be loaded on at startup
     */
    public void setLoadOnDevices(String loadOnDevices) {
        this.loadOnDevices = loadOnDevices;
    }
}
