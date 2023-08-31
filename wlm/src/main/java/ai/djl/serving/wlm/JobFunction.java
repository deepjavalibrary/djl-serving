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

import ai.djl.translate.TranslateException;

import java.util.List;

/**
 * A function describing the action to take in a {@link Job}.
 *
 * @param <I> the job input type
 * @param <O> the job output type
 */
@FunctionalInterface
public interface JobFunction<I, O> {

    /**
     * Applies this function.
     *
     * @param inputs the batch of inputs to run
     * @return the batch of results
     * @throws TranslateException if it fails to run
     */
    List<O> apply(List<I> inputs) throws TranslateException;
}
