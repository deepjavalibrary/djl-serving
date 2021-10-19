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

import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.workflow.Workflow;
import java.util.Objects;

/** A class represent a loaded {@link Workflow} and it's metadata. */
public class WorkflowInfo {

    private String name;
    private String version;
    private String modelUrl;
    private Workflow workflow;

    /**
     * Constructs a new {@link WorkflowInfo} for a single model workflow.
     *
     * @param workflowName the name of the model that will be used as HTTP endpoint
     * @param version the version of the model
     * @param modelUrl the model url
     * @param modelInfo the model to use in the workflow
     */
    public WorkflowInfo(String workflowName, String version, String modelUrl, ModelInfo modelInfo) {
        this.name = workflowName;
        this.version = version;
        this.modelUrl = modelUrl;
        this.workflow = new Workflow(modelInfo);
    }

    /**
     * Returns the workflow.
     *
     * @return the workflow
     */
    public Workflow getWorkflow() {
        return workflow;
    }

    /**
     * Returns the workflow name.
     *
     * @return the workflow name
     */
    public String getName() {
        return name;
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
        if (!(o instanceof WorkflowInfo)) {
            return false;
        }
        WorkflowInfo p = (WorkflowInfo) o;
        return name.equals(p.getName()) && version.equals(p.getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public int hashCode() {
        return Objects.hash(name, getVersion());
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        if (version != null) {
            return name + ':' + version;
        }
        return name;
    }
}
