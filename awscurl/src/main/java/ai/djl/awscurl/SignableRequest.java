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

import com.google.gson.JsonElement;
import com.google.gson.JsonPrimitive;

import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Request object for signing. */
public class SignableRequest {

    private String serviceName;
    private String httpMethod;
    private URI uri;
    private String path;
    private Map<String, String> headers;
    private Map<String, String> signedHeaders;
    private Map<String, List<String>> parameters;
    private AwsV4Signer signer;

    private byte[] content;
    private int timeOffset;
    private long signingTime;
    private int inputTokens = -1;

    /**
     * Constructs a new {@code SignableRequest} instance.
     *
     * @param serviceName the AWS service name
     * @param uri the URI of the request
     */
    public SignableRequest(String serviceName, URI uri) {
        this.serviceName = serviceName;
        httpMethod = "POST";
        headers = new ConcurrentHashMap<>();
        parameters = new LinkedHashMap<>();
        headers.put("User-Agent", "awscurl/1.0.0");
        headers.put("Accept", "*/*");
        signedHeaders = new ConcurrentHashMap<>();
        setUri(uri);
    }

    private SignableRequest() {}

    /** Signs the request. */
    public void sign() {
        if (signer == null) {
            return;
        }
        long now = System.nanoTime();
        if (now - signingTime > 2 * 60000000000L) {
            signingTime = now;
            signer.sign(this);
        }
    }

    /**
     * Makes a copy of the request.
     *
     * @return a copy of the request
     */
    public SignableRequest copy() {
        SignableRequest req = new SignableRequest();
        req.serviceName = serviceName;
        req.uri = uri;
        req.httpMethod = httpMethod;
        req.path = path;
        req.headers = new ConcurrentHashMap<>(headers);
        req.headers.remove("X-Amzn-SageMaker-Custom-Attributes");
        req.signedHeaders = new ConcurrentHashMap<>();
        req.parameters = new ConcurrentHashMap<>(parameters);
        req.signer = signer;
        req.content = content;
        req.timeOffset = timeOffset;
        return req;
    }

    /**
     * Returns the AWS service name.
     *
     * @return the AWS service name
     */
    public String getServiceName() {
        return serviceName;
    }

    /**
     * Returns the signer.
     *
     * @param signer the signer
     */
    public void setSigner(AwsV4Signer signer) {
        this.signer = signer;
    }

    private void setUri(URI uri) {
        this.uri = uri;
        String schema = uri.getScheme().toLowerCase(Locale.ENGLISH);
        int port = uri.getPort();
        String host = null;
        if (port != -1) {
            int defaultPort;
            if ("https".equals(schema)) {
                defaultPort = 443;
            } else if ("http".equals(schema)) {
                defaultPort = 80;
            } else {
                defaultPort = -1;
            }
            if (port != defaultPort) {
                host = uri.getHost() + ':' + port;
            }
        }
        if (host == null) {
            host = uri.getHost();
        }
        path = uri.getPath();
        parameters = HttpClient.parseQueryString(uri.getQuery());
        headers.put("Host", host);
    }

    /**
     * Returns the uri.
     *
     * @return the uri
     */
    public URI getUri() {
        return uri;
    }

    /**
     * Returns the HTTP method.
     *
     * @return the HTTP method
     */
    public String getHttpMethod() {
        return httpMethod;
    }

    /**
     * Sets the HTTP method.
     *
     * @param httpMethod the HTTP method
     */
    public void setHttpMethod(String httpMethod) {
        this.httpMethod = httpMethod;
    }

    /**
     * Returns the URL path.
     *
     * @return the URL path
     */
    public String getPath() {
        return path;
    }

    /**
     * Sets the HTTP headers.
     *
     * @param headers the HTTP headers
     */
    public void setHeaders(Map<String, String> headers) {
        for (Map.Entry<String, String> entry : headers.entrySet()) {
            addHeader(entry.getKey(), entry.getValue());
        }
    }

    void addHeader(String name, String value) {
        for (String key : headers.keySet()) {
            if (key.equalsIgnoreCase(name)) {
                headers.remove(key);
                break;
            }
        }
        headers.put(name, value);
    }

    /**
     * Returns the HTTP headers.
     *
     * @return the HTTP headers
     */
    public Map<String, String> getHeaders() {
        return headers;
    }

    /**
     * Returns the signed HTTP headers.
     *
     * @return the signed HTTP headers
     */
    public Map<String, String> getSignedHeaders() {
        return signedHeaders;
    }

    /**
     * Sets the signed HTTP headers.
     *
     * @param signedHeaders the signed HTTP headers
     */
    public void setSignedHeaders(Map<String, String> signedHeaders) {
        this.signedHeaders = signedHeaders;
    }

    /**
     * Sets the HTTP parameters.
     *
     * @param parameters the HTTP parameters
     */
    public void setParameters(Map<String, List<String>> parameters) {
        this.parameters.clear();
        this.parameters.putAll(parameters);
    }

    /**
     * Adds a HTTP parameter.
     *
     * @param name the parameter name
     * @param value the parameter value
     */
    public void addParameter(String name, String value) {
        List<String> paramList = parameters.computeIfAbsent(name, k -> new ArrayList<>());
        paramList.add(value);
    }

    /**
     * Adds a HTTP parameter.
     *
     * @param name the parameter name
     * @param values the parameter values
     */
    public void addParameters(String name, List<String> values) {
        for (String value : values) {
            addParameter(name, value);
        }
    }

    /**
     * Returns the HTTP parameters.
     *
     * @return the HTTP parameters
     */
    public Map<String, List<String>> getParameters() {
        return parameters;
    }

    /**
     * Returns {@code true} if has no content.
     *
     * @return {@code true} if has no content
     */
    public boolean notHasContent() {
        return content == null || content.length == 0;
    }

    /**
     * Returns the content.
     *
     * @return the content
     */
    public byte[] getContent() {
        if (content == null) {
            content = new byte[0];
        }
        return content;
    }

    /**
     * Returns the number of input tokens.
     *
     * @return the number of input tokens
     */
    public int getInputTokens() {
        if (content == null) {
            return 0;
        }

        if (inputTokens == -1) {
            boolean isJson = false;
            for (Map.Entry<String, String> entry : headers.entrySet()) {
                if ("Content-Type".equalsIgnoreCase(entry.getKey())) {
                    if ("application/json".equalsIgnoreCase(entry.getValue())) {
                        isJson = true;
                        break;
                    }
                }
            }
            if (isJson) {
                String text = new String(content, StandardCharsets.UTF_8);
                Input input = JsonUtils.GSON.fromJson(text, Input.class);
                if (input == null) {
                    return 0;
                }
                JsonElement inputs = input.inputs;
                List<String> list = new ArrayList<>();
                if (inputs != null) {
                    if (inputs.isJsonArray()) {
                        for (JsonElement element : inputs.getAsJsonArray()) {
                            String str = extractJsonString(element);
                            if (str != null) {
                                list.add(str);
                            }
                        }
                    } else {
                        String str = extractJsonString(inputs);
                        if (str != null) {
                            list.add(str);
                        }
                    }
                    inputTokens = TokenUtils.countTokens(list);
                }
            } else {
                inputTokens = 0;
            }
        }
        return inputTokens;
    }

    private String extractJsonString(JsonElement element) {
        if (element.isJsonPrimitive()) {
            JsonPrimitive primitive = element.getAsJsonPrimitive();
            if (primitive.isString()) {
                return primitive.getAsString();
            }
        }
        return null;
    }

    /**
     * Sets the content.
     *
     * @param content the content
     */
    public void setContent(byte[] content) {
        this.content = content;
    }

    /**
     * Returns the time offset.
     *
     * @return the time offset
     */
    public int getTimeOffset() {
        return timeOffset;
    }

    /**
     * Sets the time offset.
     *
     * @param timeOffset the time offset
     */
    public void setTimeOffset(int timeOffset) {
        this.timeOffset = timeOffset;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        StringBuilder builder = new StringBuilder();
        builder.append(uri.toString()).append(' ');
        if (!getHeaders().isEmpty()) {
            builder.append("Headers: (");
            for (String key : getHeaders().keySet()) {
                String value = getHeaders().get(key);
                builder.append(key).append(": ").append(value).append(", ");
            }
            builder.append(')');
        }

        return builder.toString();
    }

    private static final class Input {

        JsonElement inputs;
    }
}
