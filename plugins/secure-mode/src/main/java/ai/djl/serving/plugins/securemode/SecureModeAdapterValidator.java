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
import ai.djl.serving.wlm.Adapter;
import ai.djl.util.Utils;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/** Validates adapter security in Secure Mode. */
public final class SecureModeAdapterValidator {

    private SecureModeAdapterValidator() {}

    /**
     * Validates an adapter before it is created.
     *
     * @param adapter the adapter to validate
     * @param <I> input type
     * @param <O> output type
     * @throws IllegalConfigurationException if validation fails
     * @throws IOException if there is an error scanning the adapter directory
     */
    public static <I, O> void validateAdapter(Adapter<I, O> adapter) throws IOException {
        if (!SecureModeUtils.isSecureMode()) {
            return;
        }

        String adapterPath = adapter.getSrc();
        validateAdapterPath(adapterPath);
    }

    /**
     * Validates an adapter path before adapter creation.
     *
     * @param adapterPath the adapter source path
     * @throws IllegalConfigurationException if validation fails
     * @throws IOException if there is an error scanning the adapter directory
     */
    public static void validateAdapterPath(String adapterPath) throws IOException {
        if (!SecureModeUtils.isSecureMode()) {
            return;
        }

        String securityControls = Utils.getenv(SecureModeUtils.SECURITY_CONTROLS_ENV_VAR);
        if (securityControls == null) {
            return;
        }

        Set<String> controls = new HashSet<>(Arrays.asList(securityControls.split("\\s*,\\s*")));
        Path adapterDir = Paths.get(adapterPath);

        if (!Files.isDirectory(adapterDir)) {
            throw new IllegalConfigurationException(
                    "Adapter path " + adapterPath + " is not a valid directory.");
        }

        // Check for custom inference scripts (model.py)
        if (controls.contains(SecureModeUtils.CUSTOM_ENTRYPOINT_CONTROL)) {
            SecureModeUtils.checkModelPy(adapterDir, "adapter at " + adapterPath);
        }

        // Check for requirements.txt
        if (controls.contains(SecureModeUtils.REQUIREMENTS_TXT_CONTROL)) {
            SecureModeUtils.checkRequirementsTxt(adapterDir, "adapter at " + adapterPath);
        }

        // Check for pickle-based files
        if (controls.contains(SecureModeUtils.PICKLE_FILES_CONTROL)) {
            SecureModeUtils.scanPickleFiles(adapterDir, "adapter at " + adapterPath);
        }
    }
}
