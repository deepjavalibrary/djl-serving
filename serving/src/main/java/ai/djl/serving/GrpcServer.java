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
package ai.djl.serving;

import ai.djl.engine.Engine;
import ai.djl.serving.util.ConfigManager;
import ai.djl.util.ClassLoaderUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URL;
import java.net.URLClassLoader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.AccessController;
import java.security.PrivilegedAction;
import java.util.List;

/** A server handles gRPC protocol. */
public abstract class GrpcServer {

    private static final Logger logger = LoggerFactory.getLogger(GrpcServer.class);

    /**
     * Starts the gPRC server.
     *
     * @throws IOException if failed to start socket listener
     */
    public abstract void start() throws IOException;

    /** Stops the gPRC server. */
    public abstract void stop();

    /**
     * Creates a new {@code GrpcServer} instance.
     *
     * @return a new {@code GrpcServer} instance
     */
    public static GrpcServer newInstance() {
        try {
            ConfigManager configManager = ConfigManager.getInstance();
            List<Path> pluginsFolders = configManager.getPluginFolder();
            URL[] urls = new URL[1];
            String fileName = "grpc-" + Engine.getDjlVersion() + ".jar";
            for (Path path : pluginsFolders) {
                Path file = path.resolve(fileName);
                if (Files.isRegularFile(file)) {
                    urls[0] = file.toUri().toURL();
                    logger.info("Found gPRC plugin: {}", file);
                    break;
                }
            }
            ClassLoader cl;
            if (urls[0] != null) {
                cl =
                        AccessController.doPrivileged(
                                (PrivilegedAction<ClassLoader>) () -> new URLClassLoader(urls));
            } else {
                cl = ClassLoaderUtils.getContextClassLoader();
            }

            return ClassLoaderUtils.initClass(
                    cl, GrpcServer.class, "ai.djl.serving.grpc.GrpcServerImpl");
        } catch (IOException e) {
            logger.error("Failed to load GrpcServer", e);
        }
        return null;
    }
}
