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

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.wlm.util.WorkerJob;

import java.net.URI;
import java.net.URISyntaxException;
import java.util.Map;
import java.util.concurrent.CompletableFuture;

/**
 * An adapter is a modification producing a variation of a model that can be used during prediction.
 */
public abstract class Adapter<I, O> {

    protected ModelInfo<I, O> modelInfo;
    protected String name;
    protected String src;
    protected boolean pin;
    protected Map<String, String> options;

    /**
     * Constructs an {@link Adapter}.
     *
     * @param name the adapter name
     * @param src the adapter source
     * @param pin whether to pin the adapter
     * @param options additional adapter options
     */
    protected Adapter(
            ModelInfo<I, O> modelInfo,
            String name,
            String src,
            boolean pin,
            Map<String, String> options) {
        this.modelInfo = modelInfo;
        this.name = name;
        this.src = src;
        this.pin = pin;
        this.options = options;
    }

    /**
     * Constructs a new {@link Adapter}.
     *
     * <p>After registration, you should call {@link #register(WorkLoadManager)}. This doesn't
     * affect the worker pool itself.
     *
     * @param modelInfo the base model for the new adapter
     * @param name the adapter name
     * @param src the adapter source
     * @param pin whether to pin the adapter
     * @param options additional adapter options
     * @return the new adapter
     */
    @SuppressWarnings("unchecked")
    public static <I, O> Adapter<I, O> newInstance(
            ModelInfo<I, O> modelInfo,
            String name,
            String src,
            boolean pin,
            Map<String, String> options) {
        // TODO Allow URL support
        try {
            URI uri = new URI(src);
            String scheme = uri.getScheme();
            if (scheme != null && !"file".equals(scheme)) {
                throw new IllegalArgumentException("URL adapters are not currently supported");
            }
        } catch (URISyntaxException ignored) {
        }

        // TODO Replace usage of class name with creating adapters by Engine.newPatch(name ,src)
        if ("PyEngine".equals(modelInfo.getEngine().getClass().getSimpleName())) {
            return (Adapter<I, O>)
                    new PyAdapter((ModelInfo<Input, Output>) modelInfo, name, src, pin, options);
        } else {
            throw new IllegalArgumentException(
                    "Adapters are only currently supported for Python models");
        }
    }

    /**
     * Constructs a new {@link Adapter}.
     *
     * <p>After registration, you should call {@link #register(WorkLoadManager)}. This doesn't
     * affect the worker pool itself.
     *
     * @param modelInfo the base model for the new adapter
     * @param name the adapter name
     * @param src the adapter source
     * @param options additional adapter options
     * @return the new adapter
     */
    public static <I, O> Adapter<I, O> newInstance(
            ModelInfo<I, O> modelInfo, String name, String src, Map<String, String> options) {
        return newInstance(modelInfo, name, src, false, options);
    }

    /**
     * Unregisters an adapter in a worker pool.
     *
     * <p>This unregisters it in the wpc for new threads and all existing threads.
     *
     * @param wlm the workflow manager to remove the adapter from
     * @param <I> the input type
     * @param <O> the output type
     * @param adapterName the adapter name
     */
    public static <I, O> CompletableFuture<O> unregister(
            String adapterName, ModelInfo<I, O> modelInfo, WorkLoadManager wlm) {
        Adapter<I, O> adapter = modelInfo.unregisterAdapter(adapterName);

        // Add the unregister adapter job to job queue.
        // Because we only support one worker thread for LoRA,
        // it would be enough to add unregister adapter job once.
        Job<I, O> job = new Job<>(modelInfo, adapter.getUnregisterAdapterInput());
        return wlm.runJob(job);
    }

    /**
     * Returns the base model info.
     *
     * @return the base model info
     */
    public ModelInfo<I, O> getModelInfo() {
        return modelInfo;
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
     * Sets the adapter name.
     *
     * @param name the adapter name
     */
    public void setName(String name) {
        this.name = name;
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
     * Sets the adapter src.
     *
     * @param src the adapter src
     */
    public void setSrc(String src) {
        this.src = src;
    }

    /**
     * Returns whether to pin the adapter.
     *
     * @return whether to pin the adapter
     */
    public boolean isPin() {
        return pin;
    }

    /**
     * Sets whether to pin the adapter.
     *
     * @param pin whether to pin the adapter
     */
    public void setPin(boolean pin) {
        this.pin = pin;
    }

    /**
     * Returns the adapter options.
     *
     * @return the adapter options
     */
    public Map<String, String> getOptions() {
        return options;
    }

    /**
     * Sets the adapter options.
     *
     * @param options the adapter options
     */
    public void setOptions(Map<String, String> options) {
        this.options = options;
    }

    /**
     * Registers this adapter in a worker pool.
     *
     * <p>This registers it in the wpc for new threads and all existing threads.
     *
     * @param wlm the workload manager
     */
    public CompletableFuture<O> register(WorkLoadManager wlm) {
        modelInfo.registerAdapter(this);

        // Add the register adapter job to job queue.
        // Because we only support one worker thread for LoRA,
        // it would be enough to add register adapter job once.
        Job<I, O> job = new Job<>(modelInfo, getRegisterAdapterInput());
        return wlm.runJob(job);
    }

    /**
     * Updates this adapter in a worker pool.
     *
     * <p>This registers it in the wpc for new threads and all existing threads.
     *
     * @param wlm the workload manager to register this adapter in
     */
    public CompletableFuture<O> update(WorkLoadManager wlm) {
        modelInfo.updateAdapter(this);

        // Add the update adapter job to job queue.
        // Because we only support one worker thread for LoRA,
        // it would be enough to add update adapter job once.
        Job<I, O> job = new Job<>(modelInfo, getUpdateAdapterInput());
        return wlm.runJob(job);
    }

    /**
     * Creates a {@link WorkerJob} to register this adapter in a {@link WorkerThread}.
     *
     * @param wpc the worker pool of the thread
     * @return the registration job
     */
    public WorkerJob<I, O> registerJob(WorkerPoolConfig<I, O> wpc) {
        Job<I, O> job = new Job<>(wpc, getRegisterAdapterInput());
        return new WorkerJob<>(job, new CompletableFuture<>());
    }

    /**
     * Creates a {@link WorkerJob} to update this adapter in a {@link WorkerThread}.
     *
     * @param wpc the worker pool of the thread
     * @return the update job
     */
    public WorkerJob<I, O> updateJob(WorkerPoolConfig<I, O> wpc) {
        Job<I, O> job = new Job<>(wpc, getUpdateAdapterInput());
        return new WorkerJob<>(job, new CompletableFuture<>());
    }

    /**
     * Creates a {@link WorkerJob} to unregister this adapter from a {@link WorkerThread}.
     *
     * @param wpc the worker pool of the thread
     * @return the unregistration job
     */
    public WorkerJob<I, O> unregisterJob(WorkerPoolConfig<I, O> wpc) {
        Job<I, O> job = new Job<>(wpc, getUnregisterAdapterInput());
        return new WorkerJob<>(job, new CompletableFuture<>());
    }

    protected abstract I getRegisterAdapterInput();

    protected abstract I getUnregisterAdapterInput();

    protected abstract I getUpdateAdapterInput();
}
