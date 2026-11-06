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

import java.io.IOException;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;

/** Resolves the directory static files are served from. */
final class WebRoot {

    static final String DEFAULT = "public";

    private static final Logger logger = LoggerFactory.getLogger(WebRoot.class);
    private static final AtomicBoolean ESCAPE_WARNED = new AtomicBoolean();

    private WebRoot() {}

    /**
     * Returns the real web root, or refuses it if a link moved it.
     *
     * <p>{@link #resolve} confines the configured value lexically, which does not follow links: a
     * symbolic link at the root itself, or at any directory above it, resolves somewhere else
     * entirely. The handler anchors its per-file containment on this root, so the root is the one
     * place that has to be checked.
     *
     * <p>The real root must be exactly where the configured value says it is, not merely somewhere
     * under the server home. Containment alone is not enough: a link may relocate the root to
     * another directory in the home, and {@code conf}, {@code logs} and {@code deps} are all in the
     * home, so the whole of {@code conf} would be served as static files. The home's own links are
     * resolved on both sides, so a server home reached through a link still works.
     *
     * @param home the model server home
     * @param webRoot the lexically confined web root
     * @return the real path of the web root
     * @throws IOException if the root does not exist, or a link moved it
     */
    static Path realRootIn(Path home, Path webRoot) throws IOException {
        Path absHome = home.toAbsolutePath().normalize();
        Path absRoot = webRoot.toAbsolutePath().normalize();
        Path realHome = absHome.toRealPath();
        Path real = absRoot.toRealPath();
        Path expected;
        try {
            expected = realHome.resolve(absHome.relativize(absRoot)).normalize();
        } catch (IllegalArgumentException e) {
            // Different roots, so the configured value was never under the home.
            throw new IOException("Web root " + absRoot + " is not under " + absHome, e);
        }
        if (!real.equals(expected)) {
            if (ESCAPE_WARNED.compareAndSet(false, true)) {
                logger.warn(
                        "Web root {} resolves to {} rather than {}, so a symbolic link moved it."
                                + " Serving the bundled console instead.",
                        absRoot,
                        real,
                        expected);
            }
            throw new IOException("Web root " + absRoot + " resolves to " + real);
        }
        return real;
    }

    /**
     * Returns the web root a configured {@code WEB_ROOT} selects, confined to the server home.
     *
     * <p>{@code WEB_ROOT} is read from {@code config.properties}, which the console API can
     * rewrite. The handler's containment checks compare against this root, so a root outside the
     * server home would make them vacuous and serve whatever directory the value named. Falls back
     * to the default rather than throwing, so a bad value serves the bundled console instead of
     * taking the server down at startup.
     *
     * <p>Lexical only, so it does not follow links; {@link #realRootIn} is what establishes that
     * the root the handler anchors on really is inside the server home.
     *
     * @param home the model server home, absolute and normalized
     * @param configured the configured {@code WEB_ROOT}
     * @return a path lexically below {@code home}
     */
    static Path resolve(Path home, String configured) {
        Path root;
        try {
            Path candidate = Paths.get(configured);
            if (candidate.isAbsolute()) {
                // WEB_ROOT names a directory in the server home, so an absolute value has no
                // meaning here. Refused rather than resolved: one equal to the home would
                // otherwise pass the containment check below and serve the home itself.
                logger.warn("WEB_ROOT {} is absolute, serving {} instead.", configured, DEFAULT);
                return home.resolve(DEFAULT);
            }
            root = home.resolve(candidate).normalize();
        } catch (InvalidPathException e) {
            logger.warn("Invalid WEB_ROOT {}, serving {} instead.", configured, DEFAULT);
            return home.resolve(DEFAULT);
        }
        if (root.equals(home)) {
            // "." and "" normalize to the home, which would serve conf, logs and deps.
            logger.warn(
                    "WEB_ROOT {} selects the model server home itself, serving {} instead.",
                    configured,
                    DEFAULT);
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
