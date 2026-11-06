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
package ai.djl.serving.plugins.staticfile;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.InvalidPathException;
import java.nio.file.Path;

/** Resolves the directory static files are served from. */
final class WebRoot {

    static final String DEFAULT = "public";

    private static final Logger logger = LoggerFactory.getLogger(WebRoot.class);

    private WebRoot() {}

    /**
     * Returns the web root a configured {@code WEB_ROOT} selects, confined to the server home.
     *
     * <p>{@code WEB_ROOT} is read from {@code config.properties}, which the console API can
     * rewrite. The handler's containment checks compare against this root, so a root outside the
     * server home would make them vacuous and serve whatever directory the value named. Falls back
     * to the default rather than throwing, so a bad value serves the bundled console instead of
     * taking the server down at startup.
     *
     * @param home the model server home, absolute and normalized
     * @param configured the configured {@code WEB_ROOT}
     * @return a directory inside {@code home}
     */
    static Path resolve(Path home, String configured) {
        Path root;
        try {
            // resolve, not Paths.get: an absolute WEB_ROOT stays absolute here and is then
            // refused, where Paths.get would silently graft it under the home.
            root = home.resolve(configured).normalize();
        } catch (InvalidPathException e) {
            logger.warn("Invalid WEB_ROOT {}, serving {} instead.", configured, DEFAULT);
            return home.resolve(DEFAULT);
        }
        if (!root.startsWith(home)) {
            logger.warn(
                    "WEB_ROOT {} resolves outside the model server home {}, serving {} instead.",
                    configured,
                    home,
                    DEFAULT);
            return home.resolve(DEFAULT);
        }
        return root;
    }
}
