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

import ai.djl.Device;
import ai.djl.ModelException;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.ModelServerListenerAdapter;

import java.io.IOException;
import java.net.URISyntaxException;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    @Override
    public void onModelLoading(ModelInfo<?, ?> model, Device device) {
        super.onModelLoading(model, device);

        if (SecureModeUtils.isSecureMode()) {
            try {
                SecureModeUtils.validateSecurity();
                SecureModeUtils.reconcileSources(model.getModelUrl());
            } catch (ModelException e) {
                throw new IllegalConfigurationException("Secure Mode check failed", e);
            } catch (IOException | URISyntaxException e) {
                throw new IllegalConfigurationException(
                        "Error while running Secure Mode checks", e);
            }
            if (model.getProperties().getProperty("option.entryPoint") == null) {
                throw new IllegalConfigurationException(
                        "In Secure Mode, option.entryPoint must be explicitly set via"
                                + " serving.properties or environment variable.");
            }
        }
    }
}
