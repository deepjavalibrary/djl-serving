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

import ai.djl.modality.Input;
import ai.djl.modality.Output;

import java.util.Map;

/** An overload of {@link Adapter} for the python engine. */
public class PyAdapter extends Adapter<Input, Output> {

    /**
     * Constructs an {@link Adapter}.
     *
     * @param name the adapter name
     * @param src the adapter src
     * @param pin whether to pin the adapter
     * @param options additional adapter options
     */
    protected PyAdapter(
            ModelInfo<Input, Output> modelInfo,
            String name,
            String src,
            boolean pin,
            Map<String, String> options) {
        super(modelInfo, name, src, pin, options);
    }

    @Override
    protected Input getRegisterAdapterInput() {
        Input input = new Input();
        input.addProperty("handler", "register_adapter");
        input.addProperty("name", name);
        input.addProperty("src", src);
        input.addProperty("pin", String.valueOf(pin));
        for (Map.Entry<String, String> entry : options.entrySet()) {
            input.add(entry.getKey(), entry.getValue());
        }
        return input;
    }

    @Override
    protected Input getUpdateAdapterInput() {
        Input input = new Input();
        input.addProperty("handler", "update_adapter");
        input.addProperty("name", name);
        input.addProperty("src", src);
        input.addProperty("pin", String.valueOf(pin));
        for (Map.Entry<String, String> entry : options.entrySet()) {
            input.add(entry.getKey(), entry.getValue());
        }
        return input;
    }

    @Override
    protected Input getUnregisterAdapterInput() {
        Input input = new Input();
        input.addProperty("handler", "unregister_adapter");
        input.addProperty("name", name);
        return input;
    }
}
