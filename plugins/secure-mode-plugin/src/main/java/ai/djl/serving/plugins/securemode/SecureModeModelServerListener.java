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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    private static final Logger LOGGER =
            LoggerFactory.getLogger(SecureModeModelServerListener.class);

    @Override
    public void onModelLoading(ModelInfo<?, ?> model, Device device) {
        super.onModelLoading(model, device);
        LOGGER.info("MODEL PROPERTIES: {}", model.getProperties());
        LOGGER.info("MODEL URL: {}", model.getModelUrl());

        if (SecureModeUtils.isSecureMode()) {
            try {
                SecureModeUtils.validateSecurity();
                SecureModeUtils.reconcileSources(model.getModelUrl());
            } catch (ModelException e) {
                // TODO figure out is this is proper for exceptions
                LOGGER.error("Secure Mode check failed: ", e);
                throw new RuntimeException(e);
            } catch (IOException e) {
                LOGGER.error("Error while running Secure Mode checks: ", e);
                throw new RuntimeException(e);
            }
            if (model.getProperties().getProperty("option.entryPoint") == null) {
                LOGGER.error(
                        "In Secure Mode, option.entryPoint must be explicitly set via"
                                + " serving.properties or environment variable.");
                throw new RuntimeException("Secure Mode check failed: entryPoint is not set");
            }
        }
    }
}
