/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.models;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.wlm.ModelInfo;
import java.util.Objects;

/** An extension of {@link ModelInfo} with additional data for serving. */
public class ServingModel {

    private ModelInfo modelInfo;
    private String version;
    private String modelUrl;

    /**
     * Constructs a new {@code ModelInfo} instance.
     *
     * @param modelName the name of the model that will be used as HTTP endpoint
     * @param version the version of the model
     * @param modelUrl the model url
     * @param model the {@link ZooModel}
     * @param queueSize the maximum request queue size
     * @param maxIdleTime the initial maximum idle time for workers.
     * @param maxBatchDelay the initial maximum delay when scaling up before giving up.
     * @param batchSize the batch size for this model.
     */
    public ServingModel(
            String modelName,
            String version,
            String modelUrl,
            ZooModel<Input, Output> model,
            int queueSize,
            int maxIdleTime,
            int maxBatchDelay,
            int batchSize) {
        this.modelInfo =
                new ModelInfo(
                        modelName,
                        version,
                        model,
                        queueSize,
                        maxIdleTime,
                        maxBatchDelay,
                        batchSize);
        this.version = version;
        this.modelUrl = modelUrl;
    }

    /**
     * Returns the model info.
     *
     * @return the model info
     */
    public ModelInfo getModelInfo() {
        return modelInfo;
    }

    /**
     * Returns the model version.
     *
     * @return the model version
     */
    public String getVersion() {
        return version;
    }

    /**
     * Returns the model url.
     *
     * @return the model url
     */
    public String getModelUrl() {
        return modelUrl;
    }

    /** {@inheritDoc} */
    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (!(o instanceof ServingModel)) {
            return false;
        }
        ServingModel sm = (ServingModel) o;
        return modelInfo.equals(sm.modelInfo) && version.equals(sm.getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(modelInfo.getModelName(), getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return modelInfo.getModelName() + ':' + version;
        }
        return modelInfo.getModelName();
    }
}
