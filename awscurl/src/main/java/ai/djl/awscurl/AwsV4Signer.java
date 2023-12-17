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

import org.apache.commons.codec.binary.Hex;

import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;

import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/** A class to generate AWSv4 signature. */
public class AwsV4Signer {

    private static final String LINE_SEPARATOR = "\n";
    private static final String AWS4_TERMINATOR = "aws4_request";
    private static final String AWS4_SIGNING_ALGORITHM = "AWS4-HMAC-SHA256";

    private static final long PRESIGN_URL_MAX_EXPIRATION_SECONDS = 60 * 60 * 24 * 7;

    private static final String X_AMZ_SECURITY_TOKEN = "X-Amz-Security-Token";
    private static final String X_AMZ_CREDENTIAL = "X-Amz-Credential";
    private static final String X_AMZ_DATE = "X-Amz-Date";
    private static final String X_AMZ_EXPIRES = "X-Amz-Expires";
    private static final String X_AMZ_SIGNED_HEADER = "X-Amz-SignedHeaders";
    private static final String X_AMZ_CONTENT_SHA256 = "x-amz-content-sha256";
    private static final String X_AMZ_SIGNATURE = "X-Amz-Signature";
    private static final String X_AMZ_ALGORITHM = "X-Amz-Algorithm";
    private static final String AUTHORIZATION = "Authorization";

    private static final String ALG_HMAC_SHA256 = "HmacSHA256";
    private static final String ALG_SHA256 = "SHA-256";

    private static final List<String> HEADERS_TO_IGNORE =
            Arrays.asList("connection", "x-amzn-trace-id", "accept", "user-agent");

    private String serviceName;
    private String regionName;
    private AwsCredentials credentials;
    private Date overriddenDate;

    /**
     * Constructs a new {@code AWS4Signer} instance.
     *
     * @param serviceName the AWS servince name
     * @param region the AWS region
     * @param credentials the credentials
     */
    public AwsV4Signer(String serviceName, String region, AwsCredentials credentials) {
        this.serviceName = serviceName;
        this.regionName = region;
        this.credentials = credentials;
    }

    /**
     * Sets the override date.
     *
     * @param overriddenDate the override date
     */
    public void setOverrideDate(Date overriddenDate) {
        this.overriddenDate = overriddenDate;
    }

    /**
     * Returns the override date.
     *
     * @return the override date
     */
    public Date getOverriddenDate() {
        return overriddenDate;
    }

    /**
     * Signs the request.
     *
     * @param request the request
     */
    public void sign(SignableRequest request) {
        if (credentials.getAWSAccessKeyId() == null) {
            return;
        }

        Map<String, String> map = new ConcurrentHashMap<>();
        String sessionToken = credentials.getSessionToken();
        if (!StringUtils.isEmpty(sessionToken)) {
            map.put(X_AMZ_SECURITY_TOKEN, sessionToken);
        }

        AwsV4SignerRequestParams signerParams =
                new AwsV4SignerRequestParams(
                        request, overriddenDate, regionName, serviceName, AWS4_SIGNING_ALGORITHM);

        map.put(X_AMZ_DATE, signerParams.getFormattedSigningDateTime());

        String contentSha256 = calculateContentHash(request);
        map.put(X_AMZ_CONTENT_SHA256, contentSha256);

        String canonicalRequest = createCanonicalRequest(request, contentSha256);
        String stringToSign = createStringToSign(canonicalRequest, signerParams);
        byte[] signingKey = deriveSigningKey(credentials, signerParams);
        byte[] signature = hmacSha256(stringToSign, signingKey);

        map.put(
                AUTHORIZATION,
                buildAuthorizationHeader(request, signature, credentials, signerParams));
        request.setSignedHeaders(map);
    }

    /**
     * Generates pre-signed request.
     *
     * @param request the original http request
     * @param credentials the credential
     * @param userSpecifiedExpirationDate the expiration date
     */
    public void presignRequest(
            SignableRequest request, AwsCredentials credentials, Date userSpecifiedExpirationDate) {
        if (credentials.getAWSAccessKeyId() == null) {
            return;
        }

        Map<String, String> map = new ConcurrentHashMap<>();
        String sessionToken = credentials.getSessionToken();
        if (!StringUtils.isEmpty(sessionToken)) {
            map.put(X_AMZ_SECURITY_TOKEN, sessionToken);
        }

        AwsV4SignerRequestParams signerRequestParams =
                new AwsV4SignerRequestParams(
                        request, overriddenDate, regionName, serviceName, AWS4_SIGNING_ALGORITHM);

        // Add the important parameters for v4 signing
        String timeStamp = signerRequestParams.getFormattedSigningDateTime();

        long expirationInSeconds = generateExpirationDate(userSpecifiedExpirationDate);
        addPreSignInformationToRequest(
                request, credentials, signerRequestParams, timeStamp, expirationInSeconds);

        String contentSha256 = calculateContentHash(request);
        String canonicalRequest = createCanonicalRequest(request, contentSha256);
        String stringToSign = createStringToSign(canonicalRequest, signerRequestParams);

        byte[] signingKey = deriveSigningKey(credentials, signerRequestParams);
        byte[] signature = hmacSha256(stringToSign, signingKey);

        request.addParameter(X_AMZ_SIGNATURE, Hex.encodeHexString(signature));
        request.setSignedHeaders(map);
    }

    private String createCanonicalRequest(SignableRequest request, String contentSha256) {
        String path = request.getPath();
        return request.getHttpMethod()
                + LINE_SEPARATOR
                + getCanonicalizedResourcePath(path)
                + LINE_SEPARATOR
                + getCanonicalizedQueryString(request)
                + LINE_SEPARATOR
                + getCanonicalizedHeaderString(request)
                + LINE_SEPARATOR
                + getSignedHeadersString(request)
                + LINE_SEPARATOR
                + contentSha256;
    }

    private String createStringToSign(
            String canonicalRequest, AwsV4SignerRequestParams signerParams) {
        return signerParams.getSigningAlgorithm()
                + LINE_SEPARATOR
                + signerParams.getFormattedSigningDateTime()
                + LINE_SEPARATOR
                + signerParams.getScope()
                + LINE_SEPARATOR
                + Hex.encodeHexString(sha256(canonicalRequest));
    }

    private byte[] deriveSigningKey(
            AwsCredentials credentials, AwsV4SignerRequestParams signerRequestParams) {
        return newSigningKey(
                credentials,
                signerRequestParams.getFormattedSigningDate(),
                signerRequestParams.getRegionName(),
                signerRequestParams.getServiceName());
    }

    private String buildAuthorizationHeader(
            SignableRequest request,
            byte[] signature,
            AwsCredentials credentials,
            AwsV4SignerRequestParams signerParams) {
        String signingCredentials = credentials.getAWSAccessKeyId() + "/" + signerParams.getScope();

        String credential = "Credential=" + signingCredentials;
        String signerHeaders = "SignedHeaders=" + getSignedHeadersString(request);
        String signatureHeader = "Signature=" + Hex.encodeHexString(signature);

        return AWS4_SIGNING_ALGORITHM
                + ' '
                + credential
                + ", "
                + signerHeaders
                + ", "
                + signatureHeader;
    }

    private void addPreSignInformationToRequest(
            SignableRequest request,
            AwsCredentials credentials,
            AwsV4SignerRequestParams signerParams,
            String timeStamp,
            long expirationInSeconds) {
        String signingCredentials = credentials.getAWSAccessKeyId() + "/" + signerParams.getScope();

        request.addParameter(X_AMZ_ALGORITHM, AWS4_SIGNING_ALGORITHM);
        request.addParameter(X_AMZ_DATE, timeStamp);
        request.addParameter(X_AMZ_SIGNED_HEADER, getSignedHeadersString(request));
        request.addParameter(X_AMZ_EXPIRES, Long.toString(expirationInSeconds));
        request.addParameter(X_AMZ_CREDENTIAL, signingCredentials);
    }

    private String getCanonicalizedHeaderString(SignableRequest request) {
        List<String> sortedHeaders = new ArrayList<>(request.getHeaders().keySet());
        sortedHeaders.sort(String.CASE_INSENSITIVE_ORDER);

        Map<String, String> requestHeaders = request.getHeaders();
        StringBuilder buffer = new StringBuilder();
        for (String header : sortedHeaders) {
            if (shouldExcludeHeaderFromSigning(header)) {
                continue;
            }
            String key = header.toLowerCase(Locale.ENGLISH);
            String value = requestHeaders.get(header);

            buffer.append(key.replaceAll("\\s+", " "));
            buffer.append(':');
            if (!StringUtils.isEmpty(value)) {
                buffer.append(value.replaceAll("\\s+", " "));
            }

            buffer.append('\n');
        }

        return buffer.toString();
    }

    private String getSignedHeadersString(SignableRequest request) {
        List<String> sortedHeaders = new ArrayList<>(request.getHeaders().keySet());
        sortedHeaders.sort(String.CASE_INSENSITIVE_ORDER);

        StringBuilder buffer = new StringBuilder();
        for (String header : sortedHeaders) {
            if (shouldExcludeHeaderFromSigning(header)) {
                continue;
            }
            if (buffer.length() > 0) {
                buffer.append(';');
            }
            buffer.append(header.toLowerCase(Locale.ENGLISH));
        }

        return buffer.toString();
    }

    private boolean shouldExcludeHeaderFromSigning(String header) {
        return HEADERS_TO_IGNORE.contains(header.toLowerCase(Locale.ENGLISH));
    }

    private String calculateContentHash(SignableRequest request) {
        byte[] buf = getRequestPayload(request);
        return Hex.encodeHexString(sha256(buf));
    }

    private long generateExpirationDate(Date expirationDate) {
        long expirationInSeconds =
                expirationDate != null
                        ? ((expirationDate.getTime() - System.currentTimeMillis()) / 1000L)
                        : PRESIGN_URL_MAX_EXPIRATION_SECONDS;

        if (expirationInSeconds > PRESIGN_URL_MAX_EXPIRATION_SECONDS) {
            throw new AssertionError("Invalid expiration time: " + expirationDate);
        }
        return expirationInSeconds;
    }

    private byte[] newSigningKey(
            AwsCredentials credentials, String dateStamp, String regionName, String serviceName) {
        byte[] kSecret = ("AWS4" + credentials.getAWSSecretKey()).getBytes(StandardCharsets.UTF_8);
        byte[] kDate = hmacSha256(dateStamp, kSecret);
        byte[] kRegion = hmacSha256(regionName, kDate);
        byte[] kService = hmacSha256(serviceName, kRegion);
        return hmacSha256(AWS4_TERMINATOR, kService);
    }

    private String getCanonicalizedQueryString(Map<String, List<String>> parameters) {
        SortedMap<String, List<String>> sorted = new TreeMap<>();
        for (Map.Entry<String, List<String>> entry : parameters.entrySet()) {
            String key = entry.getKey();
            String encodedParamName = URLEncoder.encode(key, StandardCharsets.UTF_8);
            List<String> paramValues = entry.getValue();
            List<String> encodedValues = new ArrayList<>(paramValues.size());
            for (String value : paramValues) {
                encodedValues.add(URLEncoder.encode(value, StandardCharsets.UTF_8));
            }

            Collections.sort(encodedValues);
            sorted.put(encodedParamName, encodedValues);
        }

        StringBuilder result = new StringBuilder();
        for (Map.Entry<String, List<String>> entry : sorted.entrySet()) {
            for (String value : entry.getValue()) {
                if (result.length() > 0) {
                    result.append('&');
                }
                result.append(entry.getKey()).append('=').append(value);
            }
        }

        return result.toString();
    }

    private String getCanonicalizedQueryString(SignableRequest request) {
        /*
         * If we're using POST and we don't have any request payload content,
         * then any request query parameters will be sent as the payload, and
         * not in the actual query string.
         */
        if (usePayloadForQueryParameters(request)) {
            return "";
        }
        return getCanonicalizedQueryString(request.getParameters());
    }

    private byte[] getRequestPayload(SignableRequest request) {
        if (usePayloadForQueryParameters(request)) {
            String encodedParameters = encodeParameters(request);
            if (encodedParameters == null) {
                return new byte[0];
            }

            return encodedParameters.getBytes(StandardCharsets.UTF_8);
        }

        return request.getContent();
    }

    private String getCanonicalizedResourcePath(String resourcePath) {
        if (resourcePath == null || resourcePath.isEmpty()) {
            return "/";
        }

        if (resourcePath.startsWith("/")) {
            return resourcePath;
        }
        return '/' + resourcePath;
    }

    private static String encodeParameters(SignableRequest request) {
        Map<String, List<String>> requestParams = request.getParameters();
        if (requestParams.isEmpty()) {
            return null;
        }

        StringBuilder sb = new StringBuilder();
        for (Map.Entry<String, List<String>> entry : requestParams.entrySet()) {
            String parameterName = entry.getKey();
            String encodedName = URLEncoder.encode(parameterName, StandardCharsets.UTF_8);
            for (String value : entry.getValue()) {
                if (sb.length() > 0) {
                    sb.append('&');
                }
                sb.append(encodedName);
                if (!StringUtils.isEmpty(value)) {
                    sb.append('=');
                    sb.append(URLEncoder.encode(value, StandardCharsets.UTF_8));
                }
            }
        }
        return sb.toString();
    }

    private static boolean usePayloadForQueryParameters(SignableRequest request) {
        return "POST".equals(request.getHttpMethod()) && request.notHasContent();
    }

    private static byte[] hmacSha256(String message, byte[] key) {
        try {
            SecretKeySpec secretKey = new SecretKeySpec(key, ALG_HMAC_SHA256);
            Mac mac = Mac.getInstance(ALG_HMAC_SHA256);
            mac.init(secretKey);
            mac.update(message.getBytes(StandardCharsets.UTF_8));
            return mac.doFinal();
        } catch (GeneralSecurityException e) {
            throw new AssertionError(e);
        }
    }

    private static byte[] sha256(String data) {
        return sha256(data.getBytes(StandardCharsets.UTF_8));
    }

    private static byte[] sha256(byte[]... buf) {
        try {
            MessageDigest md = MessageDigest.getInstance(ALG_SHA256);
            for (byte[] b : buf) {
                md.update(b);
            }
            return md.digest();
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }
}
