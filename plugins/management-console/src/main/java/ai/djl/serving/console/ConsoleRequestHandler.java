/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.console;

import ai.djl.engine.Engine;
import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.InternalServerException;
import ai.djl.serving.http.MethodNotAllowedException;
import ai.djl.serving.http.ResourceNotFoundException;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.plugins.DependencyManager;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.MutableClassLoader;
import ai.djl.serving.util.NettyUtils;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;
import io.netty.handler.codec.http.multipart.DefaultHttpDataFactory;
import io.netty.handler.codec.http.multipart.FileUpload;
import io.netty.handler.codec.http.multipart.HttpDataFactory;
import io.netty.handler.codec.http.multipart.HttpPostRequestDecoder;
import io.netty.handler.codec.http.multipart.InterfaceHttpData;
import io.netty.handler.codec.http.multipart.InterfaceHttpData.HttpDataType;
import io.netty.handler.codec.http.multipart.MixedAttribute;
import io.netty.util.internal.StringUtil;

import org.apache.commons.compress.utils.Charsets;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Stream;

/** A class handling inbound HTTP requests for the console API. */
public class ConsoleRequestHandler implements RequestHandler<Void> {

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (!(msg instanceof HttpRequest)) {
            return false;
        }

        HttpRequest req = (HttpRequest) msg;
        return req.uri().startsWith("/console/api/");
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        switch (segments[3]) {
            case "logs":
                if (segments.length == 4) {
                    listLogs(ctx, req.method());
                } else if (segments.length == 5) {
                    int lines = NettyUtils.getIntParameter(decoder, "lines", 200);
                    showLog(ctx, segments[4], lines, req.method());
                } else if (segments.length == 6 && "download".equals(segments[4])) {
                    downloadLog(ctx, segments[5], req.method());
                } else {
                    throw new ResourceNotFoundException();
                }
                return null;
            case "inferenceAddress":
                getInferenceAddress(ctx, req.method());
                return null;
            case "upload":
                upload(ctx, req);
                return null;
            case "version":
                getVersion(ctx, req.method());
                return null;
            case "restart":
                restart(ctx, req.method());
                return null;
            case "config":
                if (HttpMethod.GET.equals(req.method())) {
                    getConfig(ctx);
                } else if (HttpMethod.POST.equals(req.method())) {
                    modifyConfig(ctx, req);
                } else {
                    throw new MethodNotAllowedException();
                }
                return null;
            case "dependency":
                if (HttpMethod.GET.equals(req.method())) {
                    listDependency(ctx);
                } else if (HttpMethod.POST.equals(req.method())) {
                    addDependency(ctx, req);
                } else if (HttpMethod.DELETE.equals(req.method())) {
                    if (segments.length == 5) {
                        deleteDependency(ctx, segments[4]);
                    } else {
                        throw new BadRequestException("Invalid url");
                    }
                } else {
                    throw new MethodNotAllowedException();
                }
                return null;
            default:
                throw new ResourceNotFoundException();
        }
    }

    private void restart(ChannelHandlerContext ctx, HttpMethod method) {
        requiresGet(method);
        Path path = Paths.get("/.dockerenv");
        if (!Files.exists(path)) {
            throw new BadRequestException("Restart is supported on Docker environment only");
        }
        NettyUtils.sendJsonResponse(
                ctx, new StatusResponse("Restart successfully, please wait a moment"));
        System.exit(333); // NOPMD
    }

    private void modifyConfig(ChannelHandlerContext ctx, FullHttpRequest req) {
        String jsonStr = req.content().toString(Charsets.toCharset("UTF-8"));
        JsonObject json = JsonParser.parseString(jsonStr).getAsJsonObject();
        String prop = json.get("prop").getAsString();
        ConfigManager configManager = ConfigManager.getInstance();
        String configFile = configManager.getProperty("configFile", "");
        try {
            Path path;
            if (!configFile.isEmpty()) {
                path = Paths.get(configFile);
            } else {
                String serverHome = ConfigManager.getModelServerHome();
                Path conf = Paths.get(serverHome, "conf");
                Files.createDirectories(conf);
                path = conf.resolve("config.properties");
            }

            Files.writeString(path, prop);
        } catch (IOException e) {
            throw new InternalServerException("Failed to write configuration file", e);
        }
        NettyUtils.sendJsonResponse(
                ctx, new StatusResponse("Configuration modification succeeded"));
    }

    private void getConfig(ChannelHandlerContext ctx) {
        ConfigManager configManager = ConfigManager.getInstance();
        String configFile = configManager.getProperty("configFile", null);
        if (configFile == null || configFile.isEmpty()) {
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(""));
            return;
        }

        Path path = Paths.get(configFile);
        try {
            String config = Files.readString(path);
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(config));
        } catch (IOException e) {
            throw new InternalServerException("Failed to read configuration file", e);
        }
    }

    private void getVersion(ChannelHandlerContext ctx, HttpMethod method) {
        requiresGet(method);
        String version = Engine.getDjlVersion();
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(version));
    }

    private void deleteDependency(ChannelHandlerContext ctx, String name) {
        if (name.contains("..")) {
            throw new BadRequestException("Invalid dependency file name:" + name);
        }
        String serverHome = ConfigManager.getModelServerHome();
        Path path = Paths.get(serverHome, "deps", name);
        try {
            Files.delete(path);
        } catch (IOException e) {
            throw new InternalServerException("Failed to delete " + name, e);
        }
        String msg = "Dependency deleted  successfully";
        NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
    }

    private void listDependency(ChannelHandlerContext ctx) {
        String serverHome = ConfigManager.getModelServerHome();
        Path depDir = Paths.get(serverHome, "deps");
        if (!Files.isDirectory(depDir)) {
            NettyUtils.sendJsonResponse(ctx, Collections.emptyList());
            return;
        }

        List<Map<String, String>> list = new ArrayList<>();
        try (Stream<Path> stream = Files.walk(depDir)) {
            stream.forEach(
                    f -> {
                        File file = f.toFile();
                        String fileName = file.getName();
                        if (fileName.endsWith(".jar")) {
                            Map<String, String> m = new ConcurrentHashMap<>(4);
                            m.put("name", fileName);
                            String[] arr = fileName.split("_");
                            if (arr.length == 3) {
                                m.put("groupId", arr[0]);
                                m.put("artifactId", arr[1]);
                                m.put("version", arr[2].replace(".jar", ""));
                            }
                            list.add(m);
                        }
                    });
        } catch (IOException e) {
            throw new InternalServerException("Failed to list dependency files", e);
        }
        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void addDependency(ChannelHandlerContext ctx, FullHttpRequest req) {
        HttpDataFactory factory = new DefaultHttpDataFactory();
        HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
        DependencyManager dm = DependencyManager.getInstance();
        try {
            List<FileUpload> fileList = new ArrayList<>();
            Map<String, String> params = new ConcurrentHashMap<>();
            List<InterfaceHttpData> bodyHttpDatas = form.getBodyHttpDatas();
            for (InterfaceHttpData data : bodyHttpDatas) {
                if (data.getHttpDataType() == HttpDataType.Attribute) {
                    MixedAttribute m = (MixedAttribute) data;
                    params.put(data.getName(), m.getValue());
                } else if (data.getHttpDataType() == HttpDataType.FileUpload) {
                    fileList.add((FileUpload) data);
                }
            }
            String type = params.getOrDefault("type", "");
            if ("engine".equals(type)) {
                String engine = params.getOrDefault("engine", "");
                dm.installEngine(engine);
            } else {
                String from = params.getOrDefault("from", "");
                if ("maven".equals(from)) {
                    String groupId = params.getOrDefault("groupId", "");
                    String artifactId = params.getOrDefault("artifactId", "");
                    String version = params.getOrDefault("version", "");
                    String dependency = groupId + ":" + artifactId + ":" + version;
                    dm.installDependency(dependency);
                } else {
                    String serverHome = ConfigManager.getModelServerHome();
                    Path depDir = Paths.get(serverHome, "deps");
                    for (FileUpload file : fileList) {
                        byte[] bytes = file.get();
                        String filename = file.getFilename();
                        Path write =
                                Files.write(
                                        Paths.get(depDir.toString(), filename),
                                        bytes,
                                        StandardOpenOption.CREATE);
                        MutableClassLoader mcl = MutableClassLoader.getInstance();
                        mcl.addURL(write.toUri().toURL());
                    }
                }
            }
            String msg = "Dependency added successfully";
            NettyUtils.sendJsonResponse(ctx, new StatusResponse(msg));
        } catch (IOException e) {
            throw new InternalServerException("Failed to install dependency", e);
        } finally {
            form.cleanFiles();
            form.destroy();
        }
    }

    private void getInferenceAddress(ChannelHandlerContext ctx, HttpMethod method) {
        requiresGet(method);
        ConfigManager configManager = ConfigManager.getInstance();
        String inferenceAddress =
                configManager.getProperty("inference_address", "http://127.0.0.1:8080");
        String managementAddress =
                configManager.getProperty("management_address", "http://127.0.0.1:8080");
        String origin = configManager.getProperty("cors_allowed_origin", "");
        String methods = configManager.getProperty("cors_allowed_methods", "");
        String headers = configManager.getProperty("cors_allowed_headers", "");
        Map<String, String> map = new ConcurrentHashMap<>(2);
        map.put("inferenceAddress", inferenceAddress);
        map.put("managementAddress", managementAddress);
        map.put("corsAllowed", "0");
        if (!StringUtil.isNullOrEmpty(origin)
                && !StringUtil.isNullOrEmpty(headers)
                && (!StringUtil.isNullOrEmpty(methods))) {
            if ("*".equals(methods) || methods.toUpperCase().contains("POST")) {
                map.put("corsAllowed", "1");
            }
        }
        NettyUtils.sendJsonResponse(ctx, map);
    }

    private void upload(ChannelHandlerContext ctx, FullHttpRequest req) {
        if (HttpPostRequestDecoder.isMultipart(req)) {
            // int sizeLimit = ConfigManager.getInstance().getMaxRequestSize();
            HttpDataFactory factory = new DefaultHttpDataFactory();
            HttpPostRequestDecoder form = new HttpPostRequestDecoder(factory, req);
            try {
                String modelServerHome = ConfigManager.getModelServerHome();
                Path dir = Paths.get(modelServerHome, "upload");
                if (!Files.isDirectory(dir)) {
                    Files.createDirectory(dir);
                }
                List<InterfaceHttpData> bodyHttpDatas = form.getBodyHttpDatas();
                InterfaceHttpData data = bodyHttpDatas.get(0);
                FileUpload fileUpload = (FileUpload) data;
                byte[] bytes = fileUpload.get();
                String filename = fileUpload.getFilename();
                Path write =
                        Files.write(
                                Paths.get(dir.toString(), filename),
                                bytes,
                                StandardOpenOption.CREATE);

                NettyUtils.sendJsonResponse(ctx, write.toUri().toString());

            } catch (IOException e) {
                throw new InternalServerException("Failed to upload file", e);
            } finally {
                form.cleanFiles();
                form.destroy();
            }
        }
    }

    private void downloadLog(ChannelHandlerContext ctx, String fileName, HttpMethod method) {
        requiresGet(method);
        String modelServerHome = ConfigManager.getModelServerHome();
        Path dir = Paths.get(modelServerHome, "logs");
        if (fileName.contains("..")) {
            throw new BadRequestException("Invalid log file name:" + fileName);
        }
        Path file = dir.resolve(fileName);
        if (!Files.isRegularFile(file)) {
            throw new BadRequestException("File does not exist");
        }
        NettyUtils.sendFile(ctx, file, true);
    }

    private void listLogs(ChannelHandlerContext ctx, HttpMethod method) {
        requiresGet(method);
        String modelServerHome = ConfigManager.getModelServerHome();
        Path dir = Paths.get(modelServerHome, "logs");
        if (!Files.isDirectory(dir)) {
            NettyUtils.sendJsonResponse(ctx, Collections.emptyList());
            return;
        }

        List<Map<String, String>> list = new ArrayList<>();
        try (Stream<Path> stream = Files.walk(dir)) {
            stream.forEach(
                    f -> {
                        File file = f.toFile();
                        String fileName = file.getName();
                        if (fileName.endsWith(".log")) {
                            Map<String, String> m = new ConcurrentHashMap<>(3);
                            m.put("name", fileName);
                            m.put("lastModified", String.valueOf(file.lastModified()));
                            m.put("length", String.valueOf(file.length()));
                            list.add(m);
                        }
                    });
        } catch (IOException e) {
            throw new InternalServerException("Failed to list log files", e);
        }
        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void showLog(ChannelHandlerContext ctx, String fileName, int lines, HttpMethod method) {
        requiresGet(method);
        String modelServerHome = ConfigManager.getModelServerHome();
        Path dir = Paths.get(modelServerHome, "logs");
        if (fileName.contains("..")) {
            throw new BadRequestException("Invalid log file name:" + fileName);
        }
        Path file = dir.resolve(fileName);
        if (!Files.isRegularFile(file)) {
            throw new BadRequestException("File does not exist");
        }

        String lastLineText = getLastLineText(file.toFile(), lines);
        NettyUtils.sendJsonResponse(ctx, lastLineText);
    }

    private String getLastLineText(File file, int lines) {
        long fileLength = file.length() - 1;
        if (fileLength < 0) {
            return "";
        }

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        try (RandomAccessFile raf = new RandomAccessFile(file, "r")) {
            int readLines = 0;
            raf.seek(fileLength);
            for (long pointer = fileLength; pointer >= 0; pointer--) {
                raf.seek(pointer);
                char c;
                c = (char) raf.read();
                if (c == '\n') {
                    readLines++;
                    if (readLines == lines) {
                        break;
                    }
                }
                bos.write(c);
                fileLength = fileLength - pointer;
            }
        } catch (IOException e) {
            throw new InternalServerException("Failed to read log file.", e);
        }
        return new String(reverse(bos.toByteArray()), StandardCharsets.UTF_8);
    }

    private static void requiresGet(HttpMethod method) {
        if (method != HttpMethod.GET) {
            throw new MethodNotAllowedException();
        }
    }

    private static byte[] reverse(byte[] array) {
        int i = 0;
        int j = array.length - 1;
        while (j > i) {
            byte tmp = array[j];
            array[j] = array[i];
            array[i] = tmp;
            j--;
            i++;
        }
        return array;
    }
}
