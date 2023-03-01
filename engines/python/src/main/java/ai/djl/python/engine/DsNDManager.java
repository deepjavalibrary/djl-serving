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
package ai.djl.python.engine;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;

/**
 * {@code DsNDManager} is the Python engine implementation of {@link NDManager}.
 *
 * <p>TODO: remove this class in 0.21.0
 */
public class DsNDManager extends PyNDManager {

    DsNDManager(Engine engine, NDManager parent, Device device) {
        super(engine, parent, device);
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine("DeepSpeed");
    }
}
