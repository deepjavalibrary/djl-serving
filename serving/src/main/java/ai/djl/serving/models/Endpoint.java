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

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/** A class that represents a webservice endpoint. */
public class Endpoint {

    private List<WorkflowInfo> workflows;
    private Map<String, Integer> map;
    private AtomicInteger position;

    /** Constructs an {@code Endpoint} instance. */
    public Endpoint() {
        workflows = new ArrayList<>();
        map = new ConcurrentHashMap<>();
        position = new AtomicInteger(0);
    }

    /**
     * Adds a workflow to the entpoint.
     *
     * @param workflow the workflow to be added
     * @return true if add success
     */
    public synchronized boolean add(WorkflowInfo workflow) {
        String version = workflow.getVersion();
        if (version == null) {
            if (workflows.isEmpty()) {
                map.put("default", 0);
                return workflows.add(workflow);
            }
            return false;
        }
        if (map.containsKey(version)) {
            return false;
        }

        map.put(version, workflows.size());
        return workflows.add(workflow);
    }

    /**
     * Returns the {@link WorkflowInfo}s associated with the endpoint.
     *
     * @return the {@link WorkflowInfo}s associated with the endpoint
     */
    public List<WorkflowInfo> getWorkflows() {
        return workflows;
    }

    /**
     * Removes a workflow version from the {@code Endpoint}.
     *
     * @param version the workflow version
     * @return null if the specified version doesn't exist
     */
    public synchronized WorkflowInfo remove(String version) {
        if (version == null) {
            if (workflows.isEmpty()) {
                return null;
            }
            WorkflowInfo workflow = workflows.remove(0);
            reIndex();
            return workflow;
        }
        Integer index = map.remove(version);
        if (index == null) {
            return null;
        }
        WorkflowInfo workflow = workflows.remove((int) index);
        reIndex();
        return workflow;
    }

    /**
     * Returns the {@code WorkflowInfo} for the specified version.
     *
     * @param version the version of the workflow to retrieve
     * @return the {@code WorkflowInfo} for the specified version
     */
    public WorkflowInfo get(String version) {
        Integer index = map.get(version);
        if (index == null) {
            return null;
        }
        return workflows.get(index);
    }

    /**
     * Returns the next version of workflow to serve the inference request.
     *
     * @return the next version of workflow to serve the inference request
     */
    public WorkflowInfo next() {
        int size = workflows.size();
        if (size == 1) {
            return workflows.get(0);
        }
        int index = position.getAndUpdate(operand -> (operand + 1) % size);
        return workflows.get(index);
    }

    private void reIndex() {
        map.clear();
        int size = workflows.size();
        for (int i = 0; i < size; ++i) {
            WorkflowInfo workflow = workflows.get(i);
            String version = workflow.getVersion();
            if (version != null) {
                map.put(version, i);
            }
        }
    }
}
