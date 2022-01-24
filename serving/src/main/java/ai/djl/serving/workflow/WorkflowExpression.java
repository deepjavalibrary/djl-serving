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

import java.util.Arrays;
import java.util.List;

/** An expression defining a local value in a {@link Workflow}. */
public class WorkflowExpression {
    private List<Item> args;

    /**
     * Constructs a {@link WorkflowExpression}.
     *
     * @param args the args to pass to the executable. Can refer to other expression value names to
     *     get the outputs of those expressions or the special name "in" to refer to the workflow
     *     input
     */
    public WorkflowExpression(Item... args) {
        this(Arrays.asList(args));
    }

    /**
     * Constructs a {@link WorkflowExpression}.
     *
     * @param args the args to pass to the executable. Can refer to other expression value names to
     *     get the outputs of those expressions or the special name "in" to refer to the workflow
     *     input
     */
    public WorkflowExpression(List<Item> args) {
        this.args = args;
    }

    /**
     * Returns the executable name (the first argument).
     *
     * @return the executable name or null if it is an expression
     */
    public String getExecutableName() {
        return args.get(0).getString();
    }

    /**
     * Returns the arguments assuming the expression is an executable (all but the first arguments).
     *
     * @return the arguments assuming the expression is an executable (all but the first arguments)
     */
    public List<Item> getExecutableArgs() {
        return args.subList(1, args.size());
    }

    /**
     * Returns the expression args.
     *
     * @return the expression args
     */
    public List<Item> getArgs() {
        return args;
    }

    /**
     * An item in the expression which contains either a string or another {@link
     * WorkflowExpression}.
     */
    public static class Item {
        private String string;
        private WorkflowExpression expression;

        /**
         * Constructs an {@link Item} containing a string.
         *
         * @param string the string
         */
        public Item(String string) {
            this.string = string;
        }

        /**
         * Constructs an {@link Item} containing a {@link WorkflowExpression}.
         *
         * @param expression the expression
         */
        public Item(WorkflowExpression expression) {
            this.expression = expression;
        }

        /**
         * Returns the string value or null if it does not contain a string.
         *
         * @return the string value or null if it does not contain a string.
         */
        public String getString() {
            return string;
        }

        /**
         * Returns the expression value or null if it does not contain an expression.
         *
         * @return the expression value or null if it does not contain an expression
         */
        public WorkflowExpression getExpression() {
            return expression;
        }
    }
}
