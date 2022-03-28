/*
 * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

/** A class that holds information about the current registered workflows. */
public class ListWorkflowsResponse {

    private String nextPageToken;
    private List<WorkflowItem> workflows;

    /** Constructs a new {@code ListWorkflowsResponse} instance. */
    public ListWorkflowsResponse() {
        workflows = new ArrayList<>();
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
     * Returns a list of workflows.
     *
     * @return a list of workflows
     */
    public List<WorkflowItem> getWorkflows() {
        return workflows;
    }

    /**
     * Adds the workflow tp the list.
     *
     * @param workflowName the workflow name
     * @param version the mode version
     */
    public void addWorkflow(String workflowName, String version) {
        workflows.add(new WorkflowItem(workflowName, version));
    }

    /** A class that holds workflow name and url. */
    public static final class WorkflowItem {

        private String workflowName;
        private String version;

        /** Constructs a new {@code WorkflowItem} instance. */
        public WorkflowItem() {}

        /**
         * Constructs a new {@code WorkflowItem} instance with workflow name and url.
         *
         * @param workflowName the workflow name
         * @param version the workflow version
         */
        public WorkflowItem(String workflowName, String version) {
            this.workflowName = workflowName;
            this.version = version;
        }

        /**
         * Returns the workflow name.
         *
         * @return the workflow name
         */
        public String getWorkflowName() {
            return workflowName;
        }

        /**
         * Returns the workflow version.
         *
         * @return the workflow version
         */
        public String getVersion() {
            return version;
        }
    }
}
