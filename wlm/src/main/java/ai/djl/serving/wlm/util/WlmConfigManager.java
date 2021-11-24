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
import ai.djl.ndarray.NDManager;

/** This manages some configurations used by the {@link ai.djl.serving.wlm.WorkLoadManager}. */
public final class WlmConfigManager {

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
        return Boolean.getBoolean("ai.djl.debug");
    }

    /**
     * Returns the default number of workers for a new registered model.
     *
     * @param manager the {@code NDManager} the model uses
     * @param deviceName the device that model loaded on
     * @param target the target number of worker
     * @return the default number of workers for a new registered model
     */
    public int getDefaultWorkers(NDManager manager, String deviceName, int target) {
        if (target == 0) {
            return 0;
        } else if (target == -1 && isDebug()) {
            return 1;
        }
        if (deviceName != null && deviceName.startsWith("nc")) {
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
}
