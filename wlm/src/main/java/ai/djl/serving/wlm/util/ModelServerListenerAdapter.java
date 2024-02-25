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

/** Base implementation of the {@link ModelServerListener} that does nothing. */
public abstract class ModelServerListenerAdapter implements ModelServerListener {

    /** {@inheritDoc} */
    @Override
    public void onModelDownloading(ModelInfo<?, ?> model) {}

    /** {@inheritDoc} */
    @Override
    public void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath) {}

    /** {@inheritDoc} */
    @Override
    public void onModelConverting(ModelInfo<?, ?> model, String type) {}

    /** {@inheritDoc} */
    @Override
    public void onModelConverted(ModelInfo<?, ?> model, String type) {}

    /** {@inheritDoc} */
    @Override
    public void onModelConfigured(ModelInfo<?, ?> model) {}

    /** {@inheritDoc} */
    @Override
    public void onModelLoading(ModelInfo<?, ?> model, Device device) {}

    /** {@inheritDoc} */
    @Override
    public void onModelLoaded(ModelInfo<?, ?> model) {}

    /** {@inheritDoc} */
    @Override
    public void onAdapterLoading(ModelInfo<?, ?> model, Path adapterPath) {}

    /** {@inheritDoc} */
    @Override
    public void onAdapterLoaded(ModelInfo<?, ?> model, Adapter adapter) {}
}
