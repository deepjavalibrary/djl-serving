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
package ai.djl.serving.wlm.util;

import ai.djl.Device;
import ai.djl.serving.wlm.Adapter;
import ai.djl.serving.wlm.ModelInfo;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** A class manages model server events. */
public final class EventManager {

    private static final EventManager INSTANCE = new EventManager();

    private List<ModelServerListener> listeners;

    private EventManager() {
        listeners = new ArrayList<>();
    }

    /**
     * Returns a singleton {@code EventManager} instance.
     *
     * @return ths {@code EventManager} instance
     */
    public static EventManager getInstance() {
        return INSTANCE;
    }

    /**
     * Adds the listener to the {@code EventManager}.
     *
     * @param listener the {@code ModelServerListener}
     */
    public void addListener(ModelServerListener listener) {
        listeners.add(listener);
    }

    /**
     * Invoked when model downloading started.
     *
     * @param model the model
     */
    public void onModelDownloading(ModelInfo<?, ?> model) {
        for (ModelServerListener l : listeners) {
            l.onModelDownloading(model);
        }
    }

    /**
     * Invoked when model downloading finished.
     *
     * @param model the model
     * @param downloadPath the model download directory
     */
    public void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath) {
        for (ModelServerListener l : listeners) {
            l.onModelDownloaded(model, downloadPath);
        }
    }

    /**
     * Invoked when model conversion started.
     *
     * @param model the model
     * @param type the conversion type
     */
    public void onModelConverting(ModelInfo<?, ?> model, String type) {
        for (ModelServerListener l : listeners) {
            l.onModelConverting(model, type);
        }
    }

    /**
     * Invoked when model conversion finished.
     *
     * @param model the model
     * @param type the conversion type
     */
    public void onModelConverted(ModelInfo<?, ?> model, String type) {
        for (ModelServerListener l : listeners) {
            l.onModelConverted(model, type);
        }
    }

    /**
     * Invoked when model properties configuration finished.
     *
     * @param model the model
     */
    public void onModelConfigured(ModelInfo<?, ?> model) {
        for (ModelServerListener l : listeners) {
            l.onModelConfigured(model);
        }
    }

    /**
     * Invoked when model loading start.
     *
     * @param model the model
     * @param device the device to load the model
     */
    public void onModelLoading(ModelInfo<?, ?> model, Device device) {
        for (ModelServerListener l : listeners) {
            l.onModelLoading(model, device);
        }
    }

    /**
     * Invoked when model loading finished.
     *
     * @param model the model
     */
    public void onModelLoaded(ModelInfo<?, ?> model) {
        for (ModelServerListener l : listeners) {
            l.onModelLoaded(model);
        }
    }

    /**
     * Invoked when adapter loading start.
     *
     * @param model the model
     * @param adapterPath the adapter path
     */
    public void onAdapterLoading(ModelInfo<?, ?> model, Path adapterPath) {
        for (ModelServerListener l : listeners) {
            l.onAdapterLoading(model, adapterPath);
        }
    }

    /**
     * Invoked when adapter loading finished.
     *
     * @param model the model
     * @param adapter the adapter
     */
    public void onAdapterLoaded(ModelInfo<?, ?> model, Adapter adapter) {
        for (ModelServerListener l : listeners) {
            l.onAdapterLoaded(model, adapter);
        }
    }
}
