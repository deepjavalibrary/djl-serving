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

import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.translate.TranslateException;

import java.util.Map;

/** An overload of {@link Adapter} for the python engine. */
public class PyAdapter extends Adapter {

    /**
     * Constructs an {@link Adapter}.
     *
     * @param name the adapter name
     * @param src the adapter src
     * @param options additional adapter options
     */
    protected PyAdapter(String name, String src, Map<String, String> options) {
        super(name, src, options);
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void registerPredictor(Predictor<?, ?> predictor) {
        Predictor<Input, Output> p = (Predictor<Input, Output>) predictor;
        Input input = new Input();
        input.addProperty("handler", "register_adapter");
        input.addProperty("name", name);
        input.addProperty("src", src);
        for (Map.Entry<String, String> entry : options.entrySet()) {
            input.add(entry.getKey(), entry.getValue());
        }
        try {
            p.predict(input);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }

    @SuppressWarnings("unchecked")
    @Override
    protected void unregisterPredictor(Predictor<?, ?> predictor) {
        Predictor<Input, Output> p = (Predictor<Input, Output>) predictor;
        Input input = new Input();
        input.addProperty("handler", "unregister_adapter");
        input.addProperty("name", name);
        try {
            p.predict(input);
        } catch (TranslateException e) {
            throw new IllegalStateException(e);
        }
    }
}
