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

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDManager;
import ai.djl.serving.wlm.ModelInfo;

/** This manages some configurations used by the {@link ai.djl.serving.wlm.WorkLoadManager}. */
public final class WlmConfigManager {

    private int jobQueueSize = 100;
    private int maxIdleTime = 60;
    private int batchSize = 1;
    private int maxBatchDelay = 300;

    private static final WlmConfigManager INSTANCE = new WlmConfigManager();

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
    public int getMaxIdleTime() {
        return maxIdleTime;
    }

    /**
     * Sets the default max idle time for workers.
     *
     * @param maxIdleTime the new default max idle time
     */
    public void setMaxIdleTime(int maxIdleTime) {
        this.maxIdleTime = maxIdleTime;
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
     * Returns the default maxBatchDelay for the working queue.
     *
     * @return the default max batch delay
     */
    public int getMaxBatchDelay() {
        return maxBatchDelay;
    }

    /**
     * Sets the default maxBatchDelay for the working queue.
     *
     * @param maxBatchDelay the new default maxBatchDelay
     */
    public void setMaxBatchDelay(int maxBatchDelay) {
        this.maxBatchDelay = maxBatchDelay;
    }

    /**
     * Returns the default minimum number of workers for a new registered model.
     *
     * @param modelInfo the {@code ModelInfo}
     * @param device the device that model loaded on
     * @param minWorkers the minimum number of workers of a new registered model
     * @param maxWorkers the maximum number of workers of a new registered model
     * @return the calculated minimum number of workers for a new registered model
     */
    public int getDefaultMinWorkers(
            ModelInfo<?, ?> modelInfo, Device device, int minWorkers, int maxWorkers) {
        if (minWorkers == 0) {
            return 0;
        }
        Model model = modelInfo.getModel(device);
        minWorkers = getWorkersProperty(model, device, "minWorkers", minWorkers);
        return Math.min(minWorkers, maxWorkers);
    }

    /**
     * Returns the default maximum number of workers for a new registered model.
     *
     * @param modelInfo the {@code ModelInfo}
     * @param device the device that model loaded on
     * @param target the target number of worker
     * @return the default number of workers for a new registered model
     */
    public int getDefaultMaxWorkers(ModelInfo<?, ?> modelInfo, Device device, int target) {
        if (target == 0) {
            return 0; // explicitly shutdown
        }

        Model model = modelInfo.getModel(device);
        if (target == -1) {
            // auto detection
            if (isDebug()) {
                return 1;
            }
            // get from model's property
            target = getWorkersProperty(model, device, "maxWorkers", -1);
            if (target > 0) {
                return target;
            }
        }

        NDManager manager = model.getNDManager();
        if (device != null && "nc".equals(device.getDeviceType())) {
            if ("Python".equals(manager.getEngine().getEngineName())) {
                return 1;
            }
            return 2; // default to max 2 workers for inferentia
        }

        if (Device.Type.GPU.equals(manager.getDevice().getDeviceType())) {
            if ("MXNet".equals(manager.getEngine().getEngineName())) {
                // FIXME: MXNet GPU Model doesn't support multi-threading
                return 1;
            } else if (target == -1) {
                target = 2; // default to max 2 workers per GPU
            }
            return target;
        }

        if (target > 0) {
            return target;
        }
        return Runtime.getRuntime().availableProcessors();
    }

    private static int getWorkersProperty(Model model, Device device, String key, int def) {
        String workers = model.getProperty(device.getDeviceType() + '.' + key);
        if (workers != null) {
            return Integer.parseInt(workers);
        }
        workers = model.getProperty(key);
        if (workers != null) {
            return Integer.parseInt(workers);
        }
        return def;
    }
}
