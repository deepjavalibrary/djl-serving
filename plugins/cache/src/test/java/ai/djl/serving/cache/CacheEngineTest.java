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

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Output;

import cloud.localstack.Localstack;
import cloud.localstack.awssdkv2.TestUtils;
import cloud.localstack.docker.annotation.LocalstackDockerAnnotationProcessor;
import cloud.localstack.docker.annotation.LocalstackDockerConfiguration;
import cloud.localstack.docker.annotation.LocalstackDockerProperties;
import cloud.localstack.docker.exception.LocalstackDockerException;

import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

import software.amazon.awssdk.services.s3.S3AsyncClient;
import software.amazon.awssdk.services.s3.S3Configuration;

import java.io.IOException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

@LocalstackDockerProperties(services = "s3")
public class CacheEngineTest {

    final byte[] buf = new byte[1];

    @Test
    public void testMemoryCacheEngine()
            throws IOException, ExecutionException, InterruptedException {
        testCacheEngine(new MemoryCacheEngine());
    }

    @Test
    public void testDdbCacheEngine() throws InterruptedException, ExecutionException, IOException {
        DynamoDbLocal server = new DynamoDbLocal();
        server.startLocalServer();
        try {
            String endpoint = "http://localhost:" + server.getPort();
            System.setProperty("DDB_ENDPOINT", endpoint);
            System.setProperty("SERVING_DDB_CACHE", "true");

            CachePlugin plugin = new CachePlugin();
            Assert.assertFalse(plugin.acceptInboundMessage(null));
            Assert.assertNull(plugin.handleRequest(null, null, null, null));

            CacheEngine engine = CacheManager.getCacheEngine();
            testCacheEngine(engine);
            Assert.assertTrue(engine instanceof DdbCacheEngine);
        } finally {
            server.stop();
            System.clearProperty("SERVING_DDB_CACHE");
            System.clearProperty("DDB_ENDPOINT");
        }
    }

    @Test
    public void testS3CacheEngine() throws IOException, ExecutionException, InterruptedException {
        Localstack localstack = Localstack.INSTANCE;
        try {
            // Start Localstack
            LocalstackDockerConfiguration dockerConfiguration =
                    new LocalstackDockerAnnotationProcessor().process(this.getClass());
            localstack.startup(dockerConfiguration);

            // Make S3AsyncClient
            S3Configuration s3Configuration =
                    S3Configuration.builder().pathStyleAccessEnabled(true).build();
            S3AsyncClient s3AsyncClient =
                    TestUtils.wrapApiClientV2(
                                    S3AsyncClient.builder(), Localstack.INSTANCE.getEndpointS3())
                            .serviceConfiguration(s3Configuration)
                            .build();

            // Make cache engine
            S3CacheEngine engine = new S3CacheEngine(false, "test-s3-cache", "", s3AsyncClient);

            // Test
            engine.createBucketIfNotExists().join();
            testCacheEngine(engine);

        } catch (IllegalStateException | LocalstackDockerException e) {
            throw new SkipException(
                    "Skipping testS3CacheEngine because it failed to start localstack");
        } finally {
            if (localstack.isRunning()) {
                localstack.stop();
            }
        }
    }

    //           Helper Functions

    private void testCacheEngine(CacheEngine engine)
            throws IOException, ExecutionException, InterruptedException {
        Assert.assertFalse(engine.isMultiTenant());

        testBasic(engine);
        testStream(engine);
    }

    private void testBasic(CacheEngine engine)
            throws ExecutionException, InterruptedException, IOException {
        // Test cache miss
        Output o = engine.get("none-exist-key", Integer.MAX_VALUE);
        Assert.assertNull(o);

        // set initial pending output
        String key1 = engine.create();
        Output pending = new Output();
        pending.addProperty("x-next-token", key1);
        CompletableFuture<Void> future = engine.put(key1, pending);
        future.get();

        // query before model generate output
        o = engine.get(key1, Integer.MAX_VALUE);
        Assert.assertEquals(o.getCode(), 200);
        Assert.assertNull(o.getData());
        String nextToken = o.getProperty("x-next-token", null);
        Assert.assertEquals(nextToken, key1);

        // retry before model generate output
        o = engine.get(nextToken, Integer.MAX_VALUE);
        Assert.assertEquals(o.getCode(), 200);

        // real output from model
        Output output1 = new Output();
        ChunkedBytesSupplier cbs1 = new ChunkedBytesSupplier();
        output1.add(cbs1);
        cbs1.appendContent(buf, true);
        future = engine.put(key1, output1);
        future.get();

        o = engine.get(nextToken, 1);
        Assert.assertEquals(o.getCode(), 200);
        Assert.assertEquals(o.getData().getAsBytes().length, 1);
        Assert.assertNull(o.getProperty("x-next-token", null));

        engine.remove(key1);
    }

    private void testStream(CacheEngine engine)
            throws IOException, ExecutionException, InterruptedException {
        String key = engine.create();
        Output output = new Output();
        output.addProperty("x-next-token", key);
        CompletableFuture<Void> future = engine.put(key, output);
        future.get();

        Output output2 = new Output();
        ChunkedBytesSupplier cbs2 = new ChunkedBytesSupplier();
        output2.add(cbs2);
        cbs2.appendContent(buf, false);
        future = engine.put(key, output2);
        for (int i = 0; i < 21; ++i) {
            cbs2.appendContent(buf, i == 20);
        }
        future.get();

        Output o = engine.get(key, 2);
        int expectedBatch = 2;
        if (engine instanceof BaseCacheEngine) {
            // 1 for initial input, 1 for write batch
            expectedBatch = 1 + ((BaseCacheEngine) engine).getWriteBatch();
        }
        Assert.assertEquals(o.getCode(), 200);
        String nextToken = o.getProperty("x-next-token", null);
        Assert.assertEquals(o.getData().getAsBytes().length, expectedBatch);
        Assert.assertNotNull(nextToken);

        o = engine.get(nextToken, 999);
        nextToken = o.getProperty("x-next-token", null);
        Assert.assertNull(nextToken);
        Assert.assertEquals(o.getData().getAsBytes().length, 22 - expectedBatch);
    }
}
