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
package ai.djl.serving.workflow;

import java.util.List;

/** An expression defining a local value in a {@link Workflow}. */
public class WorkflowExpression {
    private String executableName;
    private List<String> args;

    /**
     * Constructs a {@link WorkflowExpression}.
     *
     * @param executableName the name of the executable (model or function) to execute
     * @param args the args to pass to the executable. Can refer to other expression value names to
     *     get the outputs of those expressions or the special name "in" to refer to the workflow
     *     input
     */
    public WorkflowExpression(String executableName, List<String> args) {
        this.executableName = executableName;
        this.args = args;
    }

    /**
     * Returns the executable name.
     *
     * @return the executable name
     */
    public String getExecutableName() {
        return executableName;
    }

    /**
     * Returns the expression args.
     *
     * @return the expression args
     */
    public List<String> getArgs() {
        return args;
    }
}
