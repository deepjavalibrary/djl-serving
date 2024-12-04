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
package ai.djl.serving.http.list;

import java.util.ArrayList;
import java.util.List;

/** A class that holds information about the current registered adapters. */
public class ListAdaptersResponse {

    private String nextPageToken;
    private List<AdapterItem> adapters;

    /** Constructs a new {@code ListModelsResponse} instance. */
    public ListAdaptersResponse() {
        adapters = new ArrayList<>();
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
     * Returns a list of adapters.
     *
     * @return a list of adapters
     */
    public List<AdapterItem> getAdapters() {
        return adapters;
    }

    /**
     * Adds the adapter to the list.
     *
     * @param name the adapter name
     * @param src the adapter source
     * @param preload whether to preload the adapter during initialization
     * @param pin whether to pin the adapter
     */
    public void addAdapter(String name, String src, boolean preload, boolean pin) {
        adapters.add(new AdapterItem(name, src, preload, pin));
    }

    /** A class that holds the adapter response. */
    public static final class AdapterItem {
        private String name;
        private String src;
        private boolean preload;
        private boolean pin;

        private AdapterItem(String name, String src, boolean preload, boolean pin) {
            this.name = name;
            this.src = src;
            this.preload = preload;
            this.pin = pin;
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
         * Returns whether to pin the adapter.
         *
         * @return whether to pin the adapter
         */
        public boolean isPin() {
            return pin;
        }
    }
}
