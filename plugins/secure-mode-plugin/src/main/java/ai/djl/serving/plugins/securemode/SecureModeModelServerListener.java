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
package ai.djl.serving.plugins.securemode;

import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.ModelServerListenerAdapter;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SecureModeModelServerListener.class);

    // private static void foo(ModelInfo<?, ?> model) {
    //     LOGGER.info("Resolving Draft Model for the Model Configuration...");
    // }

    @Override
    public void onModelDownloaded(ModelInfo<?, ?> model, Path downloadPath) {
        super.onModelDownloaded(model, downloadPath);
        LOGGER.info("MODEL PROPERTIES: {}", model.getProperties());
        LOGGER.info("MODEL URL: {}", model.getModelUrl());
    }
}
