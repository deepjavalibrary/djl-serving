/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.http;

import java.util.ArrayList;
import java.util.List;

/** A class that holds information about the current registered models. */
public class ListModelsResponse {

    private String nextPageToken;
    private List<ModelItem> models;

    /** Constructs a new {@code ListModelsResponse} instance. */
    public ListModelsResponse() {
        models = new ArrayList<>();
    }

    /**
     * Returns the next page token.
     *
     * @return the next page token
     */
    public String getNextPageToken() {
        return nextPageToken;
    }

    /**
     * Sets the next page token.
     *
     * @param nextPageToken the next page token
     */
    public void setNextPageToken(String nextPageToken) {
        this.nextPageToken = nextPageToken;
    }

    /**
     * Returns a list of models.
     *
     * @return a list of models
     */
    public List<ModelItem> getModels() {
        return models;
    }

    /**
     * Adds the model tp the list.
     *
     * @param modelName the model name
     * @param version the mode version
     * @param modelUrl the model url
     * @param status the model loading status
     */
    public void addModel(String modelName, String version, String modelUrl, String status) {
        models.add(new ModelItem(modelName, version, modelUrl, status));
    }

    /** A class that holds model name and url. */
    public static final class ModelItem {

        private String modelName;
        private String version;
        private String modelUrl;
        private String status;

        /** Constructs a new {@code ModelItem} instance. */
        public ModelItem() {}

        /**
         * Constructs a new {@code ModelItem} instance with model name and url.
         *
         * @param modelName the model name
         * @param version the model version
         * @param modelUrl the model url
         * @param status the model loading status
         */
        public ModelItem(String modelName, String version, String modelUrl, String status) {
            this.modelName = modelName;
            this.version = version;
            this.modelUrl = modelUrl;
            this.status = status;
        }

        /**
         * Returns the model name.
         *
         * @return the model name
         */
        public String getModelName() {
            return modelName;
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

        /**
         * Returns the model loading status.
         *
         * @return the model loading status
         */
        public String getStatus() {
            return status;
        }
    }
}
