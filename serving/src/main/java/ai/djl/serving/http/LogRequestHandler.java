/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.http;

import ai.djl.ModelException;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.NettyUtils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.net.URLEncoder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** A class handling inbound HTTP requests for the log API. */
public class LogRequestHandler extends HttpRequestHandler {

    private static final Logger logger = LoggerFactory.getLogger(LogRequestHandler.class);
    private static final String LOG_PATH = "/logs";

    private RequestParser requestParser;

    private static final Pattern PATTERN = Pattern.compile("^/logs([/?].*)?");

    /** default constructor. */
    public LogRequestHandler() {
        this.requestParser = new RequestParser();
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) throws Exception {
        if (super.acceptInboundMessage(msg)) {
            FullHttpRequest req = (FullHttpRequest) msg;
            return PATTERN.matcher(req.uri()).matches();
        }
        return false;
    }

    /** {@inheritDoc} */
    @Override
    protected void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments)
            throws ModelException {
        HttpMethod method = req.method();
        if (!HttpMethod.GET.equals(method)) throw new MethodNotAllowedException();
        String modelServerHome = ConfigManager.getModelServerHome();
        String logsPath = modelServerHome + LOG_PATH;
        Path dir = Paths.get(logsPath);
        if (!Files.isDirectory(dir)) throw new ModelException("Log directory does not exist");
        if (segments.length < 3) {
            handleLogList(ctx, dir);
        } else if (segments.length == 3) {
            String fileName = segments[2];
            handleLogText(ctx, fileName, decoder, dir);
        } else if (segments.length == 4 && "download".equals(segments[2])) {
            String fileName = segments[3];
            handleLogDownload(ctx, dir, fileName);
        }
    }

    private void handleLogDownload(ChannelHandlerContext ctx, Path dir, String fileName)
            throws ModelException {

        File[] files =
                dir.toFile()
                        .listFiles(
                                (f, name) -> {
                                    if (name.equals(fileName)) return true;
                                    return false;
                                });
        if (files.length == 0) throw new ModelException("File does not exist");
        try {
            FileInputStream input = new FileInputStream(files[0]);
            int size = (int) files[0].length();
            byte[] data = new byte[size];
            if (size > 0) {
                int offset;
                int read;
                for (offset = 0;
                        offset < size && (read = input.read(data, offset, size - offset)) != -1;
                        offset += read) {}
            }
            HttpResponseStatus status = HttpResponseStatus.OK;
            FullHttpResponse resp =
                    new DefaultFullHttpResponse(HttpVersion.HTTP_1_1, status, false);
            String showName = URLEncoder.encode(fileName, "UTF-8");
            resp.headers()
                    .set("Content-Type", "text/plain")
                    .set(
                            "Content-Disposition",
                            "attachment;fileName=" + showName + ";fileName*=UTF-8''" + showName);
            resp.content().writeBytes(data);
            NettyUtils.sendHttpResponse(ctx, resp, true);
        } catch (IOException e) {
            logger.error("Failed to read log file", e);
            throw new ModelException("Failed to read log file");
        }
    }

    private void handleLogList(ChannelHandlerContext ctx, Path dir) {
        File[] files =
                dir.toFile()
                        .listFiles(
                                (f, name) -> {
                                    if (name.endsWith(".log")) return true;
                                    return false;
                                });
        List<Map<String, Object>> list =
                Arrays.stream(files)
                        .map(
                                v -> {
                                    Map<String, Object> m = new HashMap<String, Object>();
                                    m.put("name", v.getName());
                                    m.put("lastModified", v.lastModified());
                                    m.put("length", v.length());
                                    return m;
                                })
                        .collect(Collectors.toList());

        NettyUtils.sendJsonResponse(ctx, list);
    }

    private void handleLogText(
            ChannelHandlerContext ctx, String fileName, QueryStringDecoder decoder, Path dir)
            throws ModelException {
        File[] files =
                dir.toFile()
                        .listFiles(
                                (f, name) -> {
                                    if (name.equals(fileName)) return true;
                                    return false;
                                });
        if (files.length == 0) throw new ModelException("File does not exist");
        File file = files[0];
        int lines = NettyUtils.getIntParameter(decoder, "lines", 200);
        String lastLineText = getLastLineText(file, lines);
        NettyUtils.sendJsonResponse(ctx, lastLineText);
    }

    private String getLastLineText(File file, int lines) throws ModelException {
        StringBuilder builder = new StringBuilder();
        RandomAccessFile randomAccessFile = null;
        try {
            int readLines = 0;
            randomAccessFile = new RandomAccessFile(file, "r");
            long fileLength = file.length() - 1;
            if (fileLength < 0) return "";
            randomAccessFile.seek(fileLength);
            for (long pointer = fileLength; pointer >= 0; pointer--) {
                randomAccessFile.seek(pointer);
                char c;
                c = (char) randomAccessFile.read();
                if (c == '\n') {
                    readLines++;
                    if (readLines == lines) break;
                }
                builder.append(c);
                fileLength = fileLength - pointer;
            }
            builder.reverse();
        } catch (FileNotFoundException e) {
            logger.error("Log file does not exist", e);
            throw new ModelException("Log file does not exist");
        } catch (IOException e) {
            logger.error("Failed to read log file", e);
            throw new ModelException("Failed to read log file");
        } finally {
            try {
                randomAccessFile.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return builder.toString();
    }
}
