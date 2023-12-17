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

import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.TimeZone;

/** The AWSv4 signning request parameters. */
public class AwsV4SignerRequestParams {

    private SignableRequest request;
    private long signingDateTimeMilli;
    private String scope;
    private String regionName;
    private String serviceName;
    private String formattedSigningDateTime;
    private String formattedSigningDate;
    private String signingAlgorithm;

    /**
     * Constructs a new {@code AWS4SignerRequestParams} instance.
     *
     * @param request the request
     * @param signingDateOverride the override data
     * @param regionNameOverride the override region
     * @param serviceName the AWS service name
     * @param signingAlgorithm the signing algorithm
     */
    public AwsV4SignerRequestParams(
            SignableRequest request,
            Date signingDateOverride,
            String regionNameOverride,
            String serviceName,
            String signingAlgorithm) {
        if (request == null) {
            throw new IllegalArgumentException("Request cannot be null");
        }
        if (signingAlgorithm == null) {
            throw new IllegalArgumentException("Signing Algorithm cannot be null");
        }

        this.request = request;
        this.signingDateTimeMilli =
                signingDateOverride != null
                        ? signingDateOverride.getTime()
                        : getSigningDate(request);

        TimeZone timeZone = TimeZone.getTimeZone("UTC");
        DateFormat dateFormat = new SimpleDateFormat("yyyyMMdd", Locale.ENGLISH);
        dateFormat.setTimeZone(timeZone);
        this.formattedSigningDate = dateFormat.format(new Date(signingDateTimeMilli));
        this.serviceName = serviceName;
        this.regionName = regionNameOverride;
        this.scope = generateScope(formattedSigningDate, serviceName, regionName);

        dateFormat = new SimpleDateFormat("yyyyMMdd'T'HHmmss'Z'", Locale.ENGLISH);
        dateFormat.setTimeZone(timeZone);
        this.formattedSigningDateTime = dateFormat.format(new Date(signingDateTimeMilli));
        this.signingAlgorithm = signingAlgorithm;
    }

    private long getSigningDate(SignableRequest request) {
        return System.currentTimeMillis() - request.getTimeOffset() * 1000L;
    }

    private String generateScope(String dateStamp, String serviceName, String regionName) {
        return dateStamp + "/" + regionName + "/" + serviceName + "/" + "aws4_request";
    }

    /**
     * Returns the request.
     *
     * @return the request to be signed
     */
    public SignableRequest getRequest() {
        return request;
    }

    /**
     * Returns the scope.
     *
     * @return the scope
     */
    public String getScope() {
        return scope;
    }

    /**
     * Returns the cached datetime string.
     *
     * @return the cached datetime string
     */
    public String getFormattedSigningDateTime() {
        return formattedSigningDateTime;
    }

    /**
     * Returns the signing datetime in millis.
     *
     * @return the signing datetime in millis
     */
    public long getSigningDateTimeMilli() {
        return signingDateTimeMilli;
    }

    /**
     * Returns the AWS region.
     *
     * @return the AWS region
     */
    public String getRegionName() {
        return regionName;
    }

    /**
     * Returns the service name.
     *
     * @return the service name
     */
    public String getServiceName() {
        return serviceName;
    }

    /**
     * Returns the cached signing date.
     *
     * @return the cached signing date
     */
    public String getFormattedSigningDate() {
        return formattedSigningDate;
    }

    /**
     * Returns the signing algorithm.
     *
     * @return the signing algorithm
     */
    public String getSigningAlgorithm() {
        return signingAlgorithm;
    }
}
