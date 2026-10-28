/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.serving.http.InternalServerException;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;
import ai.djl.util.Utils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.io.IOException;
import java.io.InputStream;
import java.net.JarURLConnection;
import java.net.URL;
import java.net.URLConnection;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicReference;
import java.util.jar.JarEntry;

/**
 * A class handles static file in "public" folder or resources classpath.
 *
 * <p>this handler exposes every file which is in /static/-folder in the classpath.
 *
 * @author erik.bamberg@web.de
 */
public class StaticFileHandler implements RequestHandler<Void> {

    private static final String WEB_ROOT =
            ConfigManager.getInstance().getProperty("WEB_ROOT", "public");
    private static final Path WEB_HOME = Paths.get(ConfigManager.getModelServerHome(), WEB_ROOT);
    private static final String RESOURCE_FOLDER = "/static";

    private static final AtomicReference<Path> REAL_WEB_HOME = new AtomicReference<>();

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (!(msg instanceof FullHttpRequest)) {
            return false;
        }

        FullHttpRequest request = (FullHttpRequest) msg;
        if (!HttpMethod.GET.equals(request.method())) {
            return false;
        }
        QueryStringDecoder decoder = new QueryStringDecoder(request.uri());
        String uri = decoder.path();
        Path file = mapToPath(uri);
        return file != null && Files.isRegularFile(file) || mapToResource(uri) != null;
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest request,
            QueryStringDecoder decoder,
            String[] segments) {
        String uri = decoder.path();
        Path path = mapToPath(uri);
        if (path != null && Files.isRegularFile(path)) {
            NettyUtils.sendFile(ctx, path, HttpUtil.isKeepAlive(request));
            return null;
        }

        URL resource = mapToResource(uri);
        Objects.requireNonNull(resource);
        try (InputStream is = Utils.openUrl(resource)) {
            ResourceInfo resourceInfo = getResourceInfo(resource);
            NettyUtils.sendFile(
                    ctx,
                    is,
                    resourceInfo.name,
                    resourceInfo.lastModified,
                    HttpUtil.isKeepAlive(request));
        } catch (IOException e) {
            throw new InternalServerException("Failed to read resource file.", e);
        }
        return null;
    }

    private Path mapToPath(String uri) {
        if (uri.endsWith("/") || uri.isEmpty()) {
            return realFileInWebHome(WEB_HOME.resolve("index.html"));
        }

        try {
            return realFileInWebHome(WEB_HOME.resolve(uri.substring(1)));
        } catch (InvalidPathException e) {
            // The decoded uri can hold a NUL; not found rather than a stack trace per request.
            return null;
        }
    }

    private static Path realFileInWebHome(Path file) {
        // Lexical first, so a probe for a file that never existed skips toRealPath.
        if (!file.normalize().startsWith(WEB_HOME.normalize()) || !Files.isRegularFile(file)) {
            return null;
        }
        try {
            // Real paths on both sides: normalize() does not follow links, and the root itself
            // may be reached through one.
            Path real = file.toRealPath();
            return real.startsWith(realWebHome()) ? real : null;
        } catch (IOException e) {
            // Includes a file or a web root that does not exist, where the caller falls back to
            // the bundled resource.
            return null;
        }
    }

    private static Path realWebHome() throws IOException {
        // Cached once found; until then the web root may not exist yet.
        Path root = REAL_WEB_HOME.get();
        if (root == null) {
            root = WEB_HOME.toRealPath();
            REAL_WEB_HOME.set(root);
        }
        return root;
    }

    private static boolean hasDotDotSegment(String uri) {
        // Both separators: a classpath directory resolves the name on the file system, where
        // Windows separates on the backslash.
        for (String segment : uri.split("[/\\\\]")) {
            if ("..".equals(segment)) {
                return true;
            }
        }
        return false;
    }

    private URL mapToResource(String uri) {
        StringBuilder sb = new StringBuilder(RESOURCE_FOLDER);
        sb.append(uri);
        if (uri.endsWith("/")) {
            sb.append("index.html");
        }
        if (hasDotDotSegment(uri)) {
            // Scanned as segments rather than resolved as a Path: this runs inside
            // acceptInboundMessage, where an exception would drop the connection.
            return null;
        }
        return getClass().getResource(sb.toString());
    }

    /**
     * Retrieve the last modified date of the connection.
     *
     * @param resourceURL the url
     * @return resourceInfo which file-information like size and modifiedDate
     * @throws IOException accessing the entry
     */
    private ResourceInfo getResourceInfo(URL resourceURL) throws IOException {
        URLConnection connection = resourceURL.openConnection();
        if (connection instanceof JarURLConnection) {
            JarURLConnection jarConnection = ((JarURLConnection) connection);
            JarEntry entry = jarConnection.getJarEntry();
            if (entry != null) {
                String name = entry.getName();
                int pos = name.lastIndexOf('/');
                if (pos >= 0) {
                    name = name.substring(pos + 1);
                }
                return new ResourceInfo(name, entry.getTime());
            }
            return new ResourceInfo("not found", 0L);
        }
        try {
            return new ResourceInfo(connection.getURL().toString(), connection.getLastModified());
        } finally {
            connection.getInputStream().close();
        }
    }

    private static final class ResourceInfo {

        String name;
        long lastModified;

        public ResourceInfo(String name, long lastModified) {
            this.name = name;
            this.lastModified = lastModified;
        }
    }
}
