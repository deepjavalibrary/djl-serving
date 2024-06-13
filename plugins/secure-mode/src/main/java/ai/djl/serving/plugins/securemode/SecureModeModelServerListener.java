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

import ai.djl.serving.http.IllegalConfigurationException;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.util.ModelServerListenerAdapter;

import java.io.IOException;

class SecureModeModelServerListener extends ModelServerListenerAdapter {

    /** {@inheritDoc} */
    @Override
    public void onModelConfigured(ModelInfo<?, ?> model) {
        try {
            SecureModeUtils.validateSecurity(model);
        } catch (IOException e) {
            throw new IllegalConfigurationException("Error while running Secure Mode checks", e);
        }
    }
}
