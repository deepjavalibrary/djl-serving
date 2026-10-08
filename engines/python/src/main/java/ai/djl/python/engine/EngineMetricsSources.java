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

import java.util.HashSet;
import java.util.Set;

/**
 * Publishes the Unix sockets on which the Python workers of a model serve the metrics of their
 * engine.
 *
 * <p>A worker is published as the model property {@code engine_metrics_socket.<label>} with the
 * socket path as value. The label is derived from the device of the model: the first live worker on
 * a device gets the device label ({@code gpu0}, {@code nc2}, {@code cpu}), further live workers on
 * the same device get {@code gpu0-1}, {@code gpu0-2} and so on, always the lowest number that is
 * free. A label is never held by two live workers, so a series that carries it can restart from
 * zero but never switches to another worker.
 */
final class EngineMetricsSources {

    static final String KEY_PREFIX = "engine_metrics_socket.";

    private static final Set<String> LABELS = new HashSet<>();

    private EngineMetricsSources() {}

    /**
     * Publishes the engine metrics socket of a worker on the model.
     *
     * @param model the model the worker belongs to
     * @param socketPath the path of the Unix socket
     * @return the label of the worker, to be passed to {@link #withdraw(Model, String)}
     */
    static String publish(Model model, String socketPath) {
        String label = acquire(deviceLabel(model.getNDManager().getDevice()));
        model.setProperty(KEY_PREFIX + label, socketPath);
        return label;
    }

    /**
     * Withdraws a worker published by {@link #publish(Model, String)} and frees its label.
     *
     * <p>Calling this again for a worker that is already withdrawn does nothing, as long as the
     * label has not been handed to another worker of the same model in the meantime. The caller
     * therefore drops the label after withdrawing it.
     *
     * @param model the model the worker belongs to
     * @param label the label returned by {@link #publish(Model, String)}
     */
    static void withdraw(Model model, String label) {
        if (label != null && model.getProperties().remove(KEY_PREFIX + label) != null) {
            release(label);
        }
    }

    /**
     * Returns the label of a device.
     *
     * @param device the device
     * @return the device type followed by the device id, or just the device type if the device has
     *     no id
     */
    static String deviceLabel(Device device) {
        int id = device.getDeviceId();
        return id < 0 ? device.getDeviceType() : device.getDeviceType() + id;
    }

    private static String acquire(String deviceLabel) {
        synchronized (LABELS) {
            String label = deviceLabel;
            for (int i = 1; LABELS.contains(label); ++i) {
                label = deviceLabel + '-' + i;
            }
            LABELS.add(label);
            return label;
        }
    }

    private static void release(String label) {
        synchronized (LABELS) {
            LABELS.remove(label);
        }
    }
}
