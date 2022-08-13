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
package ai.djl.serving.http;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.nio.ByteBuffer;

/**
 * This class represents the KServe inputs and output structure for KServe infer and model request.
 */
class KServeTensor {

    String name;
    String dataType;
    long[] shape;
    // TODO: How to handle [ [ 1, 2 ], [ 4, 5 ] ] format?
    double[] data;

    public KServeTensor() {}

    public KServeTensor(String name, long[] shape, DataType type) {
        this.name = name;
        this.shape = shape;
        this.dataType = toKServeDataType(type);
    }

    static DataType fromKServeDataType(String type) {
        switch (type) {
            case "BOOL":
                return DataType.BOOLEAN;
            case "INT8":
                return DataType.UINT8;
                // TODO:
            default:
                throw new IllegalArgumentException("Invalid KServe data type: " + type);
        }
    }

    static String toKServeDataType(DataType type) {
        switch (type) {
            case BOOLEAN:
                return "BOOL";
            case INT8:
                return "INT8";
                // TODO:
            default:
                return type.toString();
        }
    }

    NDArray toTensor(NDManager manager) {
        Shape tensorShape = new Shape(shape);
        DataType type = fromKServeDataType(dataType);

        ByteBuffer bb = toByteBuffer(manager, tensorShape, type);
        NDArray array = manager.create(bb, tensorShape, type);
        array.setName(name);
        return array;
    }

    static KServeTensor fromTensor(NDArray array) {
        KServeTensor tensor = new KServeTensor();
        tensor.name = array.getName();
        tensor.dataType = toKServeDataType(array.getDataType());
        tensor.shape = array.getShape().getShape();
        Number[] values = array.toArray();
        tensor.data = new double[values.length];
        for (int i = 0; i < values.length; ++i) {
            tensor.data[i] = values[i].doubleValue();
        }

        return tensor;
    }

    private ByteBuffer toByteBuffer(NDManager manager, Shape tensorShape, DataType type) {
        int size = Math.toIntExact(tensorShape.size()) * type.getNumOfBytes();
        ByteBuffer bb = manager.allocateDirect(size);
        append(bb, data, type);
        bb.rewind();
        return bb;
    }

    private void append(ByteBuffer bb, double[] values, DataType type) {
        for (double d : values) {
            switch (type) {
                case INT8:
                    bb.put((byte) d);
                    break;
                case INT32:
                    bb.putInt((int) d);
                    break;
                case INT64:
                    bb.putLong((long) d);
                    break;
                default:
                    throw new AssertionError("Unsupported dataType: " + type);
            }
        }
    }
}
