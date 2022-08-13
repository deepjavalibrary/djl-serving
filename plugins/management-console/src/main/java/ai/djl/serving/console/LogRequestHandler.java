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

import ai.djl.serving.http.BadRequestException;
import ai.djl.serving.http.InternalServerException;
import ai.djl.serving.http.MethodNotAllowedException;
import ai.djl.serving.http.ResourceNotFoundException;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.QueryStringDecoder;

import java.io.File;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** A class handling inbound HTTP requests for the log API. */
public class LogRequestHandler implements RequestHandler<Void> {

    private static final Pattern PATTERN = Pattern.compile("^/logs([/?].*)?");

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        if (!(msg instanceof FullHttpRequest)) {
            return false;
        }

        FullHttpRequest req = (FullHttpRequest) msg;
        return PATTERN.matcher(req.uri()).matches();
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        if (!HttpMethod.GET.equals(req.method())) {
            throw new MethodNotAllowedException();
        }
        String modelServerHome = ConfigManager.getModelServerHome();
        Path dir = Paths.get(modelServerHome, "logs");
        if (segments.length < 3) {
            listLogs(ctx, dir);
        } else if (segments.length <= 4) {
            String fileName = segments[2];
            if (segments.length == 4 && "download".equals(segments[3])) {
                downloadLog(ctx, dir, fileName);
            } else {
                int lines = NettyUtils.getIntParameter(decoder, "lines", 200);
                showLog(ctx, dir, fileName, lines);
            }
        } else {
            throw new ResourceNotFoundException();
        }
        return null;
    }

    private void downloadLog(ChannelHandlerContext ctx, Path dir, String fileName) {
        if (fileName.contains("..")) {
            throw new BadRequestException("Invalid log file name:" + fileName);
        }
        Path file = dir.resolve(fileName);
        if (!Files.isRegularFile(file)) {
            throw new BadRequestException("File does not exist");
        }
        NettyUtils.sendFile(ctx, file, true);
    }

    private void listLogs(ChannelHandlerContext ctx, Path dir) {
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

    private void showLog(ChannelHandlerContext ctx, Path dir, String fileName, int lines) {
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
        if (file.length() < 0) {
            return "";
        }

        StringBuilder builder = new StringBuilder();
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
                builder.append(c);
                fileLength = fileLength - pointer;
            }
            builder.reverse();
        } catch (IOException e) {
            throw new InternalServerException("Failed to read log file.", e);
        }
        return builder.toString();
    }
}
