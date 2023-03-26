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

package ai.djl.tritonserver.engine;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.engine.StandardCapabilities;
import ai.djl.ndarray.NDManager;
import ai.djl.util.Platform;
import ai.djl.util.passthrough.PassthroughNDManager;

import org.bytedeco.tritonserver.tritonserver.TRITONSERVER_Server;

/** The {@code TsEngine} is an implementation of the {@link Engine} that runs TritonServer. */
public final class TsEngine extends Engine {

    public static final String ENGINE_NAME = "TritonServer";
    static final int RANK = 10;

    private TRITONSERVER_Server triton;

    private TsEngine(TRITONSERVER_Server triton) {
        this.triton = triton;
    }

    static Engine newInstance() {
        return new TsEngine(JniUtils.initTritonServer());
    }

    /** {@inheritDoc} */
    @Override
    public Engine getAlternativeEngine() {
        return null;
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
        Platform platform = Platform.detectPlatform("tritonserver");
        return platform.getVersion();
    }

    /** {@inheritDoc} */
    @Override
    public boolean hasCapability(String capability) {
        return StandardCapabilities.CUDA.equals(capability);
    }

    /** {@inheritDoc} */
    @Override
    public Model newModel(String name, Device device) {
        return new TsModel(name, newBaseManager(device), triton);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager() {
        return newBaseManager(null);
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newBaseManager(Device device) {
        return new PassthroughNDManager(this, device);
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return getEngineName()
                + ':'
                + getVersion()
                + ", "
                + getEngineName()
                + ':'
                + getVersion()
                + ", capabilities: [\n\t"
                + StandardCapabilities.CUDA
                + "\n]";
    }
}
