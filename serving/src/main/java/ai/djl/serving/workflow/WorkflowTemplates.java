/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.serving.workflow;

import ai.djl.util.ClassLoaderUtils;

import java.io.IOException;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** A class for managing and using {@link WorkflowDefinition} templates. */
public final class WorkflowTemplates {

    private static final Map<String, URI> TEMPLATES = new ConcurrentHashMap<>();

    private WorkflowTemplates() {}

    /**
     * Registers a new workflow template.
     *
     * @param name the template name
     * @param template the template location
     */
    public static void register(String name, URI template) {
        TEMPLATES.put(name, template);
    }

    /**
     * Constructs a {@link WorkflowDefinition} using a registered template.
     *
     * @param templateName the template name
     * @param templateReplacements a map of replacements to be applied to the template
     * @return the new {@link WorkflowDefinition} based off the template
     * @throws IOException if it fails to load the template file for parsing
     */
    public static WorkflowDefinition template(
            String templateName, Map<String, String> templateReplacements) throws IOException {
        URI uri = TEMPLATES.get(templateName);

        if (uri == null) {
            URL fromResource =
                    ClassLoaderUtils.getResource("workflowTemplates/" + templateName + ".json");
            if (fromResource != null) {
                try {
                    uri = fromResource.toURI();
                } catch (URISyntaxException ignored) {
                }
            }
        }

        if (uri == null) {
            throw new IllegalArgumentException(
                    "The workflow template " + templateName + " could not be found");
        }

        return WorkflowDefinition.parse(null, uri, templateReplacements);
    }
}
