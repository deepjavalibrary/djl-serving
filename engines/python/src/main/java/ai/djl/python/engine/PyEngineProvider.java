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
package ai.djl.python.engine;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;

/** {@code PyEngineProvider} is the Python implementation of {@link EngineProvider}. */
public class PyEngineProvider implements EngineProvider {

    private static final String ENGINE_NAME = "Python";

    private volatile Engine engine; // NOPMD
    private volatile boolean initialized; // NOPMD
    protected boolean mpiMode;

    /** {@inheritDoc} */
    @Override
    public String getEngineName() {
        return ENGINE_NAME;
    }

    /** {@inheritDoc} */
    @Override
    public int getEngineRank() {
        return PyEngine.RANK;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        if (!initialized) {
            synchronized (this) {
                if (!initialized) {
                    initialized = true;
                    PyEnv.init();
                    engine = new PyEngine(getEngineName(), mpiMode);
                }
            }
        }
        return engine;
    }
}
