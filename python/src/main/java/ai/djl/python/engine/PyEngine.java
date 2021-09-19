/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.SymbolBlock;
import ai.djl.training.GradientCollector;

/** The {@code PyEngine} is an implementation of the {@link Engine} that runs Python worker. */
public final class PyEngine extends Engine {

    public static final String ENGINE_NAME = "Python";
    static final int RANK = 10;

    private Engine alternativeEngine;
    private boolean initialized;

    private PyEngine() {}

    static Engine newInstance() {
        PyEnv.init();
        return new PyEngine();
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        if (!initialized && !Boolean.getBoolean("ai.djl.python.disable_alternative")) {
            Engine engine = Engine.getInstance();
            if (engine.getRank() < getRank()) {
                // alternativeEngine should not have the same rank as OnnxRuntime
                alternativeEngine = engine;
            }
            initialized = true;
        }
        return alternativeEngine;
    }

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getRank() {
        return RANK;
    }

    /** {@inheritDoc} */
    @Override
    public String getVersion() {
        return PyEnv.getVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return true;
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new PyModel(name, newBaseManager(device));
    }

    /** {@inheritDoc} */
    @Override
    public SymbolBlock newSymbolBlock(NDManager manager) {
        throw new UnsupportedOperationException("Python engine does not support SymbolBlock");
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return PyNDManager.getSystemManager().newSubManager(device);
    }

    /** {@inheritDoc} */
    @Override
    public GradientCollector newGradientCollector() {
        throw new UnsupportedOperationException("Not supported for Python engine");
    }

    /** {@inheritDoc} */
    @Override
    public void setRandomSeed(int seed) {
        throw new UnsupportedOperationException("Not supported for Python engine");
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName() + ':' + getVersion();
    }
}
