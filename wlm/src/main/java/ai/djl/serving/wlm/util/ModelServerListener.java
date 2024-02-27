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

/** An interface that represent a model server event listener. */
public interface ModelServerListener {

    /**
     * Invoked when model downloading started.
     *
     * @param model the model
     */
    void onModelDownloading(ModelInfo<?, ?> model);

    /**
     * Invoked when model downloading finished.
     *
     * @param model the model
     * @param downloadPath the model download directory
     */
    void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath);

    /**
     * Invoked when model conversion started.
     *
     * @param model the model
     * @param type the conversion type
     */
    void onModelConverting(ModelInfo<?, ?> model, String type);

    /**
     * Invoked when model conversion finished.
     *
     * @param model the model
     * @param type the conversion type
     */
    void onModelConverted(ModelInfo<?, ?> model, String type);

    /**
     * Invoked when model properties configuration finished.
     *
     * @param model the model
     */
    void onModelConfigured(ModelInfo<?, ?> model);

    /**
     * Invoked when model loading start.
     *
     * @param model the model
     * @param device the device to load the model
     */
    void onModelLoading(ModelInfo<?, ?> model, Device device);

    /**
     * Invoked when model loading finished.
     *
     * @param model the model
     */
    void onModelLoaded(ModelInfo<?, ?> model);

    /**
     * Invoked when adapter loading start.
     *
     * @param model the model
     * @param adapterPath the adapter path
     */
    void onAdapterLoading(ModelInfo<?, ?> model, Path adapterPath);

    /**
     * Invoked when adapter loading finished.
     *
     * @param model the model
     * @param adapter the adapter
     */
    void onAdapterLoaded(ModelInfo<?, ?> model, Adapter adapter);
}
