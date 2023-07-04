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
import ai.djl.inference.streaming.PublisherBytesSupplier;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** A helper for creating a {@link CacheEngine}. */
public abstract class BaseCacheEngine implements CacheEngine {

    private static final Logger logger = LoggerFactory.getLogger(BaseCacheEngine.class);
    protected int writeBatch;

    /** Creates a {@link BaseCacheEngine}. */
    public BaseCacheEngine() {
        writeBatch = Integer.parseInt(Utils.getenv("SERVING_CACHE_BATCH", "1"));
    }

    /** {@inheritDoc} */
    @Override
    public Output get(String key, int limit) {
        int start = 0;
        if (key.length() > 36) {
            start = Integer.parseInt(key.substring(36));
            key = key.substring(0, 36);
        }
        return get(key, start, limit);
    }

    protected abstract Output get(String key, int start, int limit);

    /** {@inheritDoc} */
    @Override
    public CompletableFuture<Void> put(String key, Output output) {
        return CompletableFuture.<Void>supplyAsync(
                        () -> {
                            try {
                                BytesSupplier supplier = output.getData();
                                if (supplier instanceof ChunkedBytesSupplier) {
                                    Output o = new Output();
                                    o.setCode(output.getCode());
                                    o.setMessage(output.getMessage());
                                    o.setProperties(output.getProperties());
                                    ChunkedBytesSupplier cbs = (ChunkedBytesSupplier) supplier;
                                    int index = 0;
                                    putStream(key, o, cbs.pollChunk(), index++, !cbs.hasNext());
                                    List<byte[]> list = new ArrayList<>(writeBatch);
                                    while (cbs.hasNext()) {
                                        try {
                                            list.add(cbs.nextChunk(1, TimeUnit.MINUTES));
                                        } catch (InterruptedException e) {
                                            throw new IllegalStateException(e);
                                        }
                                        if (list.size() >= writeBatch) {
                                            byte[] batch = joinBytes(list);
                                            putStream(key, null, batch, index++, !cbs.hasNext());
                                            list.clear();
                                        }
                                    }
                                    if (!list.isEmpty()) {
                                        byte[] batch = joinBytes(list);
                                        putStream(key, null, batch, index, true);
                                    }
                                } else if (supplier instanceof PublisherBytesSupplier) {
                                    Output o = new Output();
                                    o.setCode(output.getCode());
                                    o.setMessage(output.getMessage());
                                    o.setProperties(output.getProperties());
                                    PublisherBytesSupplier pub = (PublisherBytesSupplier) supplier;
                                    AtomicInteger index = new AtomicInteger(-1);
                                    List<byte[]> list = new ArrayList<>(writeBatch);
                                    putStream(key, o, null, index.incrementAndGet(), false);
                                    pub.subscribe(
                                            buf -> {
                                                try {
                                                    if (buf == null) {
                                                        byte[] batch = joinBytes(list);
                                                        putStream(
                                                                key,
                                                                null,
                                                                batch,
                                                                index.incrementAndGet(),
                                                                true);
                                                    } else if (buf.length > 0) {
                                                        list.add(buf);
                                                        if (list.size() >= writeBatch) {
                                                            byte[] batch = joinBytes(list);
                                                            putStream(
                                                                    key,
                                                                    null,
                                                                    batch,
                                                                    index.incrementAndGet(),
                                                                    false);
                                                            list.clear();
                                                        }
                                                    }
                                                } catch (IOException e) {
                                                    throw new CompletionException(e);
                                                }
                                            });
                                } else {
                                    boolean last = output.getCode() != 202;
                                    putSingle(key, output, last);
                                }
                            } catch (IOException e) {
                                throw new CompletionException(e);
                            }
                            return null;
                        })
                .exceptionally(
                        t -> {
                            logger.warn("Failed to write to Cache", t);
                            return null;
                        });
    }

    /**
     * Returns the number of elements to batch before putting with {@link #putStream(String, Output,
     * byte[], int, boolean)}.
     *
     * @return number of elements to batch
     */
    public int getWriteBatch() {
        return writeBatch;
    }

    protected abstract void putSingle(String key, Output output, boolean last) throws IOException;

    protected abstract void putStream(
            String key, Output output, byte[] buf, int index, boolean last) throws IOException;

    protected byte[] joinBytes(List<byte[]> list) {
        return joinBytes(list, -1);
    }

    protected byte[] joinBytes(List<byte[]> list, int size) {
        if (list.size() == 1) {
            return list.get(0);
        }
        if (size < 0) {
            size = 0;
            for (byte[] buf : list) {
                size += buf.length;
            }
        }
        byte[] batch = new byte[size];
        size = 0;
        for (byte[] buf : list) {
            System.arraycopy(buf, 0, batch, size, buf.length);
            size += buf.length;
        }
        return batch;
    }
}
