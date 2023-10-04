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
package ai.djl.serving.wlm;

import ai.djl.inference.Predictor;
import ai.djl.serving.wlm.WorkerPoolConfig.ThreadConfig;
import ai.djl.serving.wlm.util.WorkerJob;
import ai.djl.util.Utils;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.concurrent.CompletableFuture;

/**
 * An adapter is a modification producing a variation of a model that can be used during prediction.
 */
public abstract class Adapter {

    protected String name;
    protected String src;

    /**
     * Constructs an {@link Adapter}.
     *
     * @param name the adapter name
     * @param src the adapter source
     */
    protected Adapter(String name, String src) {
        this.name = name;
        this.src = src;
    }

    /**
     * Constructs a new {@link Adapter}.
     *
     * <p>After registration, you should call {@link #register(WorkerPool)}. This doesn't affect the
     * worker pool itself.
     *
     * @param wpc the worker pool config for the new adapter
     * @param name the adapter name
     * @param src the adapter source
     * @return the new adapter
     */
    public static Adapter newInstance(WorkerPoolConfig<?, ?> wpc, String name, String src) {
        if (!Boolean.parseBoolean(Utils.getEnvOrSystemProperty("ENABLE_ADAPTERS_PREVIEW"))) {
            throw new IllegalStateException("Adapters preview is not enabled");
        }

        if (!(wpc instanceof ModelInfo)) {
            String modelName = wpc.getId();
            throw new IllegalArgumentException("The worker " + modelName + " is not a model");
        }

        // TODO Allow URL support
        try {
            URI uri = new URI(src);
            String scheme = uri.getScheme();
            if (scheme != null && !"file".equals(scheme)) {
                throw new IllegalArgumentException("URL adapters are not currently supported");
            }
        } catch (URISyntaxException ignored) {
        }

        ModelInfo<?, ?> modelInfo = (ModelInfo<?, ?>) wpc;
        // TODO Replace usage of class name with creating adapters by Engine.newPatch(name ,src)
        if ("PyEngine".equals(modelInfo.getEngine().getClass().getSimpleName())) {
            return new PyAdapter(name, src);
        } else {
            throw new IllegalArgumentException(
                    "Adapters are only currently supported for Python models");
        }
    }

    /**
     * Unregisters an adapter in a worker pool.
     *
     * <p>This unregisters it in the wpc for new threads and all existing threads.
     *
     * @param wp the worker pool to remove the adapter from
     * @param adapterName the adapter name
     * @param <I> the input type
     * @param <O> the output type
     */
    public static <I, O> void unregister(WorkerPool<I, O> wp, String adapterName) {
        ModelInfo<I, O> wpc = (ModelInfo<I, O>) wp.getWpc();
        Adapter adapter = wpc.unregisterAdapter(adapterName);
        // TODO Support worker adapter scheduling rather than register/unregister on all workers
        for (WorkerGroup<I, O> wg : wp.getWorkerGroups().values()) {
            for (WorkerThread<I, O> t : wg.getWorkers()) {
                t.addConfigJob(adapter.unregisterJob(wpc, t.getThreadType()));
            }
        }
    }

    /**
     * Returns the adapter name.
     *
     * @return the adapter name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the adapter src.
     *
     * @return the adapter src
     */
    public String getSrc() {
        return src;
    }

    /**
     * Registers this adapter in a worker pool.
     *
     * <p>This registers it in the wpc for new threads and all existing threads.
     *
     * @param wp the worker pool to register this adapter in
     * @param <I> the input type
     * @param <O> the output type
     */
    public <I, O> void register(WorkerPool<I, O> wp) {
        ModelInfo<I, O> wpc = (ModelInfo<I, O>) wp.getWpc();
        wpc.registerAdapter(this);
        for (WorkerGroup<I, O> wg : wp.getWorkerGroups().values()) {
            for (WorkerThread<I, O> t : wg.getWorkers()) {
                t.addConfigJob(registerJob(wpc, t.getThreadType()));
            }
        }
    }

    /**
     * Creates a {@link WorkerJob} to register this adapter in a {@link WorkerThread}.
     *
     * @param wpc the worker pool of the thread
     * @param threadConfig the thread config to register
     * @param <I> the input type
     * @param <O> the output type
     * @return the registration job
     */
    public <I, O> WorkerJob<I, O> registerJob(
            WorkerPoolConfig<I, O> wpc, ThreadConfig<I, O> threadConfig) {
        ModelInfo<I, O>.ModelThread t = (ModelInfo<I, O>.ModelThread) threadConfig;
        Job<I, O> job =
                new Job<>(
                        wpc,
                        null,
                        in -> {
                            registerPredictor(t.getPredictor());
                            return null;
                        });
        return new WorkerJob<>(job, new CompletableFuture<>());
    }

    /**
     * Creates a {@link WorkerJob} to unregister this adapter from a {@link WorkerThread}.
     *
     * @param wpc the worker pool of the thread
     * @param threadConfig the thread config to unregister
     * @param <I> the input type
     * @param <O> the output type
     * @return the unregistration job
     */
    public <I, O> WorkerJob<I, O> unregisterJob(
            WorkerPoolConfig<I, O> wpc, ThreadConfig<I, O> threadConfig) {
        ModelInfo<I, O>.ModelThread t = (ModelInfo<I, O>.ModelThread) threadConfig;
        Job<I, O> job =
                new Job<>(
                        wpc,
                        null,
                        in -> {
                            unregisterPredictor(t.getPredictor());
                            return null;
                        });
        return new WorkerJob<>(job, new CompletableFuture<>());
    }

    protected abstract void registerPredictor(Predictor<?, ?> predictor);

    protected abstract void unregisterPredictor(Predictor<?, ?> predictor);
}
