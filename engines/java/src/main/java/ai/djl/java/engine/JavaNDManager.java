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
package ai.djl.java.engine;

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

/** {@code JavaNDManager} is the Java engine implementation of {@link NDManager}. */
public class JavaNDManager extends BaseNDManager {

    private static final JavaNDManager SYSTEM_MANAGER = new SystemManager();

    private JavaNDManager(NDManager parent, Device device) {
        super(parent, device);
    }

    static JavaNDManager getSystemManager() {
        return SYSTEM_MANAGER;
    }

    /** {@inheritDoc} */
    @Override
    public ByteBuffer allocateDirect(int capacity) {
        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder());
    }

    /** {@inheritDoc} */
    @Override
    public JavaNDArray from(NDArray array) {
        if (array == null || array instanceof JavaNDArray) {
            return (JavaNDArray) array;
        }
        return (JavaNDArray) create(array.toByteBuffer(), array.getShape(), array.getDataType());
    }

    /** {@inheritDoc} */
    @Override
    public NDManager newSubManager(Device device) {
        JavaNDManager manager = new JavaNDManager(this, device);
        attachInternal(manager.uid, manager);
        return manager;
    }

    /** {@inheritDoc} */
    @Override
    public Engine getEngine() {
        return Engine.getEngine(JavaEngine.ENGINE_NAME);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray create(Buffer data, Shape shape, DataType dataType) {
        int size = Math.toIntExact(shape.size());
        BaseNDManager.validateBufferSize(data, dataType, size);
        if (data instanceof ByteBuffer) {
            return new JavaNDArray(this, alternativeManager, (ByteBuffer) data, shape, dataType);
        }
        ByteBuffer bb = ByteBuffer.allocate(size * dataType.getNumOfBytes());
        bb.order(ByteOrder.nativeOrder());
        BaseNDManager.copyBuffer(data, bb);
        return new JavaNDArray(this, alternativeManager, bb, shape, dataType);
    }

    /** The SystemManager is the root {@link JavaNDManager} of which all others are children. */
    private static final class SystemManager extends JavaNDManager {

        SystemManager() {
            super(null, null);
        }

        /** {@inheritDoc} */
        @Override
        public void attachInternal(String resourceId, AutoCloseable resource) {}

        /** {@inheritDoc} */
        @Override
        public void detachInternal(String resourceId) {}

        /** {@inheritDoc} */
        @Override
        public void close() {}
    }
}
