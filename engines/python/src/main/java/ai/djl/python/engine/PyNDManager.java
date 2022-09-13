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

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.BaseNDManager;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** {@code PyNDManager} is the Python engine implementation of {@link NDManager}. */
public class PyNDManager extends BaseNDManager {

    private static final PyNDManager SYSTEM_MANAGER = new SystemManager();

    private PyNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static PyNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public PyNDArray from(NDArray array) {
        if (array == null || array instanceof PyNDArray) {
            return (PyNDArray) array;
        }
        return create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public PyNDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBuffer(data, dataType, size);
        if (data instanceof ByteBuffer) {
            return new PyNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
        }

        ByteBuffer bb = ByteBuffer.allocate(size * dataType.getNumOfBytes());
        bb.order(ByteOrder.nativeOrder());
        BaseNDManager.copyBuffer(data, bb);
        return new PyNDArray(this, alternativeManager, bb, shape, dataType);
    }

    /** {@inheritDoc} */
    @Override
    public PyNDManager newSubManager(Device device) {
        PyNDManager manager = new PyNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public final Engine getEngine() {
        return Engine.getEngine(PyEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public void close() {
        super.close();
        if (alternativeManager != null) {
            alternativeManager.close();
            alternativeManager = null;
        }
    }

    /** The SystemManager is the root {@link PyNDManager} of which all others are children. */
    private static final class SystemManager extends PyNDManager implements SystemNDManager {

        SystemManager() {
            super(null, null);
        }

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
