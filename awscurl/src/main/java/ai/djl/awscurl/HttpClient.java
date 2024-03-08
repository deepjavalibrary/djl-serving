/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.awscurl;

import ai.djl.util.Utils;

import com.google.gson.JsonElement;
import com.google.gson.JsonParseException;

import org.apache.http.Header;
import org.apache.http.HttpResponse;
import org.apache.http.HttpVersion;
import org.apache.http.StatusLine;
import org.apache.http.client.config.RequestConfig;
import org.apache.http.client.methods.HttpDelete;
import org.apache.http.client.methods.HttpEntityEnclosingRequestBase;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpHead;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.client.methods.HttpPut;
import org.apache.http.client.methods.HttpUriRequest;
import org.apache.http.conn.ssl.NoopHostnameVerifier;
import org.apache.http.conn.ssl.SSLConnectionSocketFactory;
import org.apache.http.conn.ssl.TrustAllStrategy;
import org.apache.http.entity.ByteArrayEntity;
import org.apache.http.impl.client.CloseableHttpClient;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.message.BasicHttpResponse;
import org.apache.http.message.BasicStatusLine;
import org.apache.http.ssl.SSLContextBuilder;

import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.URI;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.atomic.AtomicInteger;

import javax.net.ssl.HostnameVerifier;
import javax.net.ssl.SSLContext;

@SuppressWarnings("PMD.SystemPrintln")
final class HttpClient {

    private HttpClient() {}

    public static HttpResponse sendRequest(
            SignableRequest request,
            boolean insecure,
            int timeout,
            OutputStream ps,
            boolean dumpHeader,
            AtomicInteger tokens,
            long[] requestTime,
            String[] names)
            throws IOException {
        ps.write(("\n" + System.currentTimeMillis() + ": ").getBytes(StandardCharsets.UTF_8));
        long begin = System.nanoTime();
        try (CloseableHttpClient client = getHttpClient(insecure, timeout)) {
            HttpUriRequest req =
                    createHttpRequest(
                            request.getHttpMethod(), request.getUri(), request.getContent());
            if (dumpHeader) {
                String path = request.getUri().getPath();
                if (!path.startsWith("/") && !path.isEmpty()) {
                    path = path.substring(1);
                }
                System.out.println("> " + request.getHttpMethod() + " /" + path + " HTTP/1.1");
                System.out.println("> ");
            }

            addHeaders(req, request.getHeaders(), dumpHeader);
            addHeaders(req, request.getSignedHeaders(), dumpHeader);

            HttpResponse resp = client.execute(req);
            int code = resp.getStatusLine().getStatusCode();
            if (dumpHeader) {
                System.out.println("> ");
                System.out.println(
                        "< HTTP/1.1 " + code + ' ' + resp.getStatusLine().getReasonPhrase());
                System.out.println("< ");
                for (Header header : resp.getAllHeaders()) {
                    System.out.println("< " + header.getName() + ": " + header.getValue());
                }
                System.out.println("< ");
            }

            if (code >= 300 && ps instanceof NullOutputStream) {
                System.out.println(
                        "HTTP error ("
                                + resp.getStatusLine()
                                + "): "
                                + Utils.toString(resp.getEntity().getContent()));
                return resp;
            }

            Header[] headers = resp.getHeaders("Content-Type");
            String contentType = null;
            if (headers != null) {
                for (Header header : headers) {
                    String[] parts = header.getValue().split(";");
                    contentType = parts[0];
                    if ("text/event-stream".equals(contentType)) {
                        break;
                    }
                }
            }

            HttpResponse ret = resp;
            try (FirstByteCounterInputStream is =
                    new FirstByteCounterInputStream(resp.getEntity().getContent())) {
                if (tokens != null) {
                    JsonUtils.resetException();
                    if (contentType == null || "text/plain".equals(contentType)) {
                        String body = Utils.toString(is);
                        ps.write(body.getBytes(StandardCharsets.UTF_8));
                        updateTokenCount(Collections.singletonList(body), tokens, request);
                    } else if ("application/json".equals(contentType)) {
                        String body = Utils.toString(is);
                        ps.write(body.getBytes(StandardCharsets.UTF_8));
                        try {
                            JsonElement element = JsonUtils.GSON.fromJson(body, JsonElement.class);
                            List<String> lines = new ArrayList<>();
                            JsonUtils.getJsonList(element, lines, names);
                            updateTokenCount(lines, tokens, request);
                        } catch (JsonParseException e) {
                            AwsCurl.logger.warn("Invalid json response: {}", body);
                            StatusLine status =
                                    new BasicStatusLine(HttpVersion.HTTP_1_1, 500, "error");
                            ret = new BasicHttpResponse(status);
                        }
                    } else if ("application/jsonlines".equals(contentType)) {
                        boolean hasError = false;
                        try (BufferedReader reader =
                                new BufferedReader(
                                        new InputStreamReader(is, StandardCharsets.UTF_8))) {
                            String line;
                            List<StringBuilder> list = new ArrayList<>();
                            while ((line = reader.readLine()) != null) {
                                hasError =
                                        JsonUtils.processJsonLine(list, ps, line, names)
                                                || hasError;
                            }
                            updateTokenCount(list, tokens, request);
                        }
                        if (hasError) {
                            StatusLine status =
                                    new BasicStatusLine(HttpVersion.HTTP_1_1, 500, "error");
                            ret = new BasicHttpResponse(status);
                        }
                    } else if ("text/event-stream".equals(contentType)) {
                        List<String> list = new ArrayList<>();
                        try (BufferedReader reader =
                                new BufferedReader(
                                        new InputStreamReader(is, StandardCharsets.UTF_8))) {
                            String line;
                            while ((line = reader.readLine()) != null) {
                                line = line.trim();
                                if (!line.startsWith("data:")) {
                                    continue;
                                }
                                if (requestTime[1] == -1) {
                                    requestTime[1] = System.nanoTime() - begin;
                                }
                                line = line.substring(5);
                                JsonElement element =
                                        JsonUtils.GSON.fromJson(line, JsonElement.class);
                                JsonUtils.getJsonList(element, list, names);
                            }
                            updateTokenCount(list, tokens, request);
                        }
                    } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                        List<StringBuilder> list = new ArrayList<>();
                        handleEventStream(is, list, names, ps);
                        updateTokenCount(list, tokens, request);
                    }
                } else if ("application/vnd.amazon.eventstream".equals(contentType)) {
                    List<StringBuilder> list = new ArrayList<>();
                    handleEventStream(is, list, names, ps);
                } else {
                    is.transferTo(ps);
                    ps.flush();
                }
                requestTime[0] += System.nanoTime() - begin;
                if (requestTime[1] == -1) {
                    requestTime[1] = is.getTimeToFirstByte() - begin;
                }
            }
            return ret;
        }
    }

    private static void handleEventStream(
            InputStream is, List<StringBuilder> list, String[] names, OutputStream ps)
            throws IOException {
        byte[] buf = new byte[12];
        byte[] payload = new byte[512];
        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        while (true) {
            if (is.readNBytes(buf, 0, buf.length) == 0) {
                break;
            }
            ByteBuffer bb = ByteBuffer.wrap(buf);
            bb.order(ByteOrder.BIG_ENDIAN);
            int totalLength = bb.getInt();
            int headerLength = bb.getInt();
            int payloadLength = totalLength - headerLength - 12 - 4;
            int size = totalLength - 12;
            if (size > payload.length) {
                payload = new byte[size];
            }
            if (is.readNBytes(payload, 0, size) == 0) {
                break;
            }
            if (payloadLength == 0) {
                break;
            }
            bos.write(payload, headerLength, payloadLength);
        }
        bos.close();

        byte[] bytes = bos.toByteArray();
        Scanner scanner = new Scanner(new ByteArrayInputStream(bytes), StandardCharsets.UTF_8);
        while (scanner.hasNext()) {
            String line = scanner.nextLine();
            if (JsonUtils.processJsonLine(list, ps, line, names)) {
                throw new IOException("Response contains error");
            }
        }
        scanner.close();
    }

    private static void addHeaders(HttpUriRequest req, Map<String, String> headers, boolean dump) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            if (dump) {
                System.out.println("> " + entry.getKey() + ": " + entry.getValue());
            }
            req.addHeader(entry.getKey(), entry.getValue());
        }
    }

    public static Map<String, List<String>> parseQueryString(String queryString) {
        Map<String, List<String>> parameters = new LinkedHashMap<>(); // NOPMD
        if (StringUtils.isEmpty(queryString)) {
            return parameters;
        }

        for (String pair : queryString.split("&")) {
            String[] parameter = pair.split("=", 2);
            List<String> list = parameters.computeIfAbsent(parameter[0], k -> new ArrayList<>());
            if (parameter.length > 1) {
                list.add(parameter[1]);
            } else {
                list.add(null);
            }
        }
        return parameters;
    }

    private static CloseableHttpClient getHttpClient(boolean insecure, int timeout) {
        RequestConfig config =
                RequestConfig.custom()
                        .setConnectTimeout(timeout)
                        .setConnectionRequestTimeout(timeout)
                        .setSocketTimeout(timeout)
                        .build();
        if (insecure) {
            try {
                SSLContext context =
                        SSLContextBuilder.create()
                                .loadTrustMaterial(TrustAllStrategy.INSTANCE)
                                .build();

                HostnameVerifier verifier = new NoopHostnameVerifier();
                SSLConnectionSocketFactory factory =
                        new SSLConnectionSocketFactory(context, verifier);

                return HttpClients.custom()
                        .setDefaultRequestConfig(config)
                        .setSSLSocketFactory(factory)
                        .build();
            } catch (GeneralSecurityException e) {
                throw new AssertionError(e);
            }
        }
        return HttpClients.custom().setDefaultRequestConfig(config).build();
    }

    private static HttpUriRequest createHttpRequest(String method, URI uri, byte[] data) {
        HttpUriRequest request;

        if (HttpPost.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpPost(uri);
        } else if (HttpPut.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpPut(uri);
        } else if (HttpDelete.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpDelete(uri);
        } else if (HttpGet.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpGet(uri);
        } else if (HttpHead.METHOD_NAME.equalsIgnoreCase(method)) {
            request = new HttpHead(uri);
        } else {
            throw new IllegalArgumentException("Invalid method: " + method);
        }

        if (request instanceof HttpEntityEnclosingRequestBase && data != null) {
            ByteArrayEntity entity = new ByteArrayEntity(data);
            ((HttpEntityEnclosingRequestBase) request).setEntity(entity);
        }

        return request;
    }

    static void updateTokenCount(
            List<? extends CharSequence> list, AtomicInteger tokens, SignableRequest request) {
        tokens.addAndGet(TokenUtils.countTokens(list));
        if (Utils.getEnvOrSystemProperty("EXCLUDE_INPUT_TOKEN") != null) {
            tokens.addAndGet(-request.getInputTokens());
        }
    }

    private static final class FirstByteCounterInputStream extends InputStream {

        private long timeToFirstByte;
        private InputStream is;

        FirstByteCounterInputStream(InputStream is) {
            this.is = is;
        }

        long getTimeToFirstByte() {
            return timeToFirstByte;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b) throws IOException {
            int read = is.read(b);
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public int read(byte[] b, int off, int len) throws IOException {
            int read = is.read(b, off, len);
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public int read() throws IOException {
            int read = is.read();
            if (timeToFirstByte == 0 && read > 0) {
                timeToFirstByte = System.nanoTime();
            }
            return read;
        }

        /** {@inheritDoc} */
        @Override
        public void close() throws IOException {
            is.close();
        }
    }
}
