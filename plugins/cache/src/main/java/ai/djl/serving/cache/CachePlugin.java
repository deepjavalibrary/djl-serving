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

import ai.djl.aws.s3.S3RepositoryFactory;
import ai.djl.repository.Repository;
import ai.djl.serving.plugins.RequestHandler;
import ai.djl.util.Utils;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import software.amazon.awssdk.core.exception.SdkClientException;

import java.util.concurrent.CompletionException;

/** A plugin handles caching options. */
public class CachePlugin implements RequestHandler<Void> {

    private static final Logger logger = LoggerFactory.getLogger(CachePlugin.class);

    /** Constructs a new {@code CachePlugin} instance. */
    public CachePlugin() {
        Repository.registerRepositoryFactory(new S3RepositoryFactory());
        boolean multiTenant =
                Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_CACHE_MULTITENANT"));
        if (Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_DDB_CACHE"))) {
            try {
                DdbCacheEngine engine = DdbCacheEngine.newInstance();
                CacheManager.setCacheEngine(engine);
                logger.info("DynamoDB cache is enabled.");
            } catch (SdkClientException e) {
                logger.warn("Failed to create DynamoDB cache", e);
            }
        } else if (Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_S3_CACHE"))) {
            try {
                String bucket = Utils.getEnvOrSystemProperty("SERVING_S3_CACHE_BUCKET");
                String keyPrefix = Utils.getEnvOrSystemProperty("SERVING_S3_CACHE_KEY_PREFIX");
                S3CacheEngine engine = new S3CacheEngine(multiTenant, bucket, keyPrefix);
                if (Boolean.parseBoolean(
                        Utils.getEnvOrSystemProperty("SERVING_S3_CACHE_AUTOCREATE"))) {
                    engine.createBucketIfNotExists().join();
                }
                CacheManager.setCacheEngine(engine);
                logger.info("S3 cache is enabled.");
            } catch (CompletionException e) {
                logger.warn("Failed to create S3 cache ", e);
            }
        } else {
            String capacity = Utils.getEnvOrSystemProperty("SERVING_MEMORY_CACHE_CAPACITY");
            MemoryCacheEngine engine;
            if (capacity != null) {
                engine = new MemoryCacheEngine(multiTenant, Integer.parseInt(capacity));
            } else {
                engine = new MemoryCacheEngine(multiTenant);
            }
            CacheManager.setCacheEngine(engine);
        }
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        return null;
    }
}
