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
package ai.djl.serving.ddbcache;

import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Output;
import ai.djl.serving.cache.CacheEngine;
import ai.djl.serving.cache.CacheManager;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;

public class DdbCacheEngineTest {

    @Test
    public void testDdbCacheEngine() throws InterruptedException, ExecutionException {
        DynamoDbLocal server = new DynamoDbLocal();
        server.startLocalServer();
        try {
            String endpoint = "http://localhost:" + server.getPort();
            System.setProperty("DDB_ENDPOINT", endpoint);
            System.setProperty("SERVING_DDB_CACHE", "true");

            DdbCachePlugin plugin = new DdbCachePlugin();
            Assert.assertFalse(plugin.acceptInboundMessage(null));
            Assert.assertNull(plugin.handleRequest(null, null, null, null));

            CacheEngine engine = CacheManager.getCacheEngine();
            Assert.assertTrue(engine instanceof DdbCacheEngine);
            Assert.assertFalse(engine.isMultiTenant());

            Output o = engine.get("none-exist-key", Integer.MAX_VALUE);
            Assert.assertNull(o);

            byte[] buf = new byte[1];

            // set initial pending output
            String key1 = engine.create();
            Output pending = new Output();
            pending.setCode(202);
            pending.addProperty("x-next-token", key1);
            CompletableFuture<Void> future = engine.put(key1, pending);
            future.get();

            // query before model generate output
            o = engine.get(key1, Integer.MAX_VALUE);
            Assert.assertEquals(o.getCode(), 200);
            Assert.assertNull(o.getData());
            String nextToken = o.getProperty("x-next-token", null);
            Assert.assertEquals(nextToken, key1 + "-1");

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

            String key2 = engine.create();
            pending.addProperty("x-next-token", key2);
            future = engine.put(key2, pending);
            future.get();

            Output output2 = new Output();
            ChunkedBytesSupplier cbs2 = new ChunkedBytesSupplier();
            output2.add(cbs2);
            cbs2.appendContent(buf, false);
            future = engine.put(key2, output2);
            for (int i = 0; i < 21; ++i) {
                cbs2.appendContent(buf, i == 20);
            }
            future.get();

            o = engine.get(key2, 2);
            Assert.assertEquals(o.getCode(), 200);
            nextToken = o.getProperty("x-next-token", null);
            Assert.assertEquals(o.getData().getAsBytes().length, 6);

            o = engine.get(nextToken, 9);
            nextToken = o.getProperty("x-next-token", null);
            Assert.assertNull(nextToken);
            Assert.assertEquals(o.getData().getAsBytes().length, 16);

            engine.remove(key1);
        } finally {
            server.stop();
            System.clearProperty("SERVING_DDB_CACHE");
            System.clearProperty("DDB_ENDPOINT");
        }
    }
}
