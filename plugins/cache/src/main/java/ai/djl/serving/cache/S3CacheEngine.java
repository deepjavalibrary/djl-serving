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
package ai.djl.serving.cache;

import ai.djl.modality.Output;

import software.amazon.awssdk.core.ResponseInputStream;
import software.amazon.awssdk.core.async.AsyncRequestBody;
import software.amazon.awssdk.core.internal.async.InputStreamResponseTransformer;
import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.model.BucketLifecycleConfiguration;
import software.amazon.awssdk.services.s3.model.CreateBucketRequest;
import software.amazon.awssdk.services.s3.model.DeleteObjectRequest;
import software.amazon.awssdk.services.s3.model.ExpirationStatus;
import software.amazon.awssdk.services.s3.model.GetObjectRequest;
import software.amazon.awssdk.services.s3.model.GetObjectResponse;
import software.amazon.awssdk.services.s3.model.HeadBucketRequest;
import software.amazon.awssdk.services.s3.model.LifecycleExpiration;
import software.amazon.awssdk.services.s3.model.LifecycleRule;
import software.amazon.awssdk.services.s3.model.LifecycleRuleFilter;
import software.amazon.awssdk.services.s3.model.ListObjectsRequest;
import software.amazon.awssdk.services.s3.model.PutBucketLifecycleConfigurationRequest;
import software.amazon.awssdk.services.s3.model.PutObjectRequest;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

/** A {@link CacheEngine} that stores elements in S3. */
public class S3CacheEngine extends BaseCacheEngine {

    S3AsyncClient asyncClient;

    boolean multiTenant;
    String bucket;
    String keyPrefix;

    /**
     * Constructs a {@link S3CacheEngine}.
     *
     * @param multiTenant whether to create separate entries for each user
     * @param bucket the S3 bucket
     */
    public S3CacheEngine(boolean multiTenant, String bucket) {
        this(multiTenant, bucket, "");
    }

    /**
     * Constructs a {@link S3CacheEngine}.
     *
     * @param multiTenant whether to create separate entries for each user
     * @param bucket the S3 bucket
     * @param keyPrefix the S3 key prefix for all cache entries
     */
    public S3CacheEngine(boolean multiTenant, String bucket, String keyPrefix) {
        this(multiTenant, bucket, keyPrefix, null);
    }

    /**
     * Constructs a {@link S3CacheEngine}.
     *
     * @param multiTenant whether to create separate entries for each user
     * @param bucket the S3 bucket
     * @param keyPrefix the S3 key prefix for all cache entries
     * @param asyncClient an S3 client (optional)
     */
    public S3CacheEngine(
            boolean multiTenant, String bucket, String keyPrefix, S3AsyncClient asyncClient) {
        this.multiTenant = multiTenant;
        this.bucket = bucket;
        this.keyPrefix = keyPrefix;

        if (bucket == null) {
            throw new IllegalStateException(
                    "When using the S3CacheEngine, the bucket can't be null or missing. Try setting"
                            + " SERVING_S3_CACHE_BUCKET.");
        }

        if (keyPrefix == null) {
            this.keyPrefix = "";
        }

        if (asyncClient == null) {
            asyncClient = S3AsyncClient.builder().build();
        }
        this.asyncClient = asyncClient;
    }

    /**
     * Creates the matching S3 bucket if it doesn't exist.
     *
     * @return a future to await the creation.
     */
    public CompletableFuture<Void> createBucketIfNotExists() {
        return bucketExists()
                .thenCompose(
                        exists -> {
                            if (exists) {
                                return null;
                            }
                            CreateBucketRequest req =
                                    CreateBucketRequest.builder().bucket(bucket).build();
                            return asyncClient
                                    .createBucket(req)
                                    .thenComposeAsync(
                                            (r) -> {
                                                LifecycleRule rule =
                                                        LifecycleRule.builder()
                                                                .id(
                                                                        "DJL Serving S3 Cache"
                                                                                + " Expiration for "
                                                                                + keyPrefix)
                                                                .filter(
                                                                        LifecycleRuleFilter
                                                                                .builder()
                                                                                .prefix(keyPrefix)
                                                                                .build())
                                                                .expiration(
                                                                        LifecycleExpiration
                                                                                .builder()
                                                                                .days(1)
                                                                                .build())
                                                                .status(ExpirationStatus.ENABLED)
                                                                .build();
                                                BucketLifecycleConfiguration
                                                        lifecycleConfiguration =
                                                                BucketLifecycleConfiguration
                                                                        .builder()
                                                                        .rules(rule)
                                                                        .build();
                                                PutBucketLifecycleConfigurationRequest
                                                        lifecycleReq =
                                                                PutBucketLifecycleConfigurationRequest
                                                                        .builder()
                                                                        .bucket(bucket)
                                                                        .lifecycleConfiguration(
                                                                                lifecycleConfiguration)
                                                                        .build();
                                                return asyncClient.putBucketLifecycleConfiguration(
                                                        lifecycleReq);
                                            })
                                    .thenAccept(r -> {});
                        });
    }

    /**
     * Returns true if the matching bucket name exists.
     *
     * @return true if the matching bucket name exists.
     */
    public CompletableFuture<Boolean> bucketExists() {
        HeadBucketRequest req = HeadBucketRequest.builder().bucket(bucket).build();
        return asyncClient.headBucket(req).handle((r, t) -> r != null);
    }

    /** {@inheritDoc} */
    @Override
    public boolean isMultiTenant() {
        return multiTenant;
    }

    /** {@inheritDoc} */
    @Override
    @SuppressWarnings({"unchecked", "rawtypes"})
    protected Output get(String key, int start, int limit) {
        ListObjectsRequest lsReq =
                ListObjectsRequest.builder().bucket(bucket).prefix(keyPrefix + key).build();
        return asyncClient
                .listObjects(lsReq)
                .thenCompose(
                        ls -> {
                            int objectCount = ls.contents().size();
                            int objectsToRequest = Math.min(objectCount - start, limit);

                            if (objectsToRequest == 0) {
                                return CompletableFuture.completedFuture(null);
                            }

                            CompletableFuture<ResponseInputStream<GetObjectResponse>>[] responses =
                                    new CompletableFuture[objectsToRequest];
                            for (int i = 0; i < objectsToRequest; i++) {
                                int iReq = start + i;
                                GetObjectRequest req =
                                        GetObjectRequest.builder()
                                                .bucket(bucket)
                                                .key(keyPrefix + key + iReq)
                                                .build();
                                responses[i] =
                                        asyncClient.getObject(
                                                req, new InputStreamResponseTransformer<>());
                            }
                            return CompletableFuture.allOf(responses)
                                    .thenApply(
                                            v -> {
                                                List<byte[]> dataToJoin =
                                                        new ArrayList<>(objectsToRequest);
                                                Output output = new Output();
                                                if (start == 0) {
                                                    try (ResponseInputStream<GetObjectResponse> is =
                                                            responses[0].join()) {
                                                        Output o = Output.decode(is);
                                                        output.setCode(o.getCode());
                                                        output.setMessage(o.getMessage());
                                                        output.setProperties(o.getProperties());
                                                        if (o.getData() != null) {
                                                            dataToJoin.add(
                                                                    o.getData().getAsBytes());
                                                        }
                                                    } catch (IOException e) {
                                                        throw new CompletionException(e);
                                                    }
                                                }

                                                boolean returnedLastItem =
                                                        Boolean.parseBoolean(
                                                                responses[objectsToRequest - 1]
                                                                        .join()
                                                                        .response()
                                                                        .metadata()
                                                                        .get("last"));
                                                if (!returnedLastItem
                                                        && !output.getProperties()
                                                                .containsKey("x-next-token")) {
                                                    output.addProperty(
                                                            "x-next-token",
                                                            key + (start + objectsToRequest));

                                                    output.addProperty(
                                                            "X-Amzn-SageMaker-Custom-Attributes",
                                                            "x-next-token="
                                                                    + key
                                                                    + (start + objectsToRequest));
                                                }

                                                for (int i = (start == 0 ? 1 : 0);
                                                        i < objectsToRequest;
                                                        i++) {
                                                    try (ResponseInputStream<GetObjectResponse> is =
                                                            responses[i].join()) {
                                                        dataToJoin.add(is.readAllBytes());
                                                    } catch (IOException e) {
                                                        throw new CompletionException(e);
                                                    }
                                                }
                                                if (!dataToJoin.isEmpty()) {
                                                    byte[] data = joinBytes(dataToJoin);
                                                    output.add(data);
                                                }

                                                return output;
                                            });
                        })
                .join();
    }

    /** {@inheritDoc} */
    @Override
    protected void putSingle(String key, Output output, boolean last) throws IOException {
        putStream(key, output, null, 0, last);
    }

    /** {@inheritDoc} */
    @Override
    protected void putStream(String key, Output output, byte[] buf, int index, boolean last)
            throws IOException {
        PutObjectRequest req =
                PutObjectRequest.builder()
                        .bucket(bucket)
                        .key(keyPrefix + key + index)
                        .metadata(Collections.singletonMap("last", Boolean.toString(last)))
                        .build();

        AsyncRequestBody body;
        if (output != null) {
            if (buf != null) {
                output.add(buf);
            }
            body = AsyncRequestBody.fromByteBuffer(ByteBuffer.wrap(output.encode()));
        } else {
            body = AsyncRequestBody.fromByteBuffer(ByteBuffer.wrap(buf));
        }

        asyncClient.putObject(req, body).thenApply(r -> null).join();
    }

    /** {@inheritDoc} */
    @Override
    public void remove(String key) {
        DeleteObjectRequest req =
                DeleteObjectRequest.builder().bucket(bucket).key(keyPrefix + key).build();
        asyncClient.deleteObject(req).join();
    }
}
