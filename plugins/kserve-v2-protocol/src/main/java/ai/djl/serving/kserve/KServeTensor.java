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
package ai.djl.serving.kserve;

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
            case "UINT8":
                return DataType.UINT8;
            case "INT8":
                return DataType.INT8;
            case "INT32":
                return DataType.INT32;
            case "INT64":
                return DataType.INT64;
            case "FP16":
                return DataType.FLOAT16;
            case "FP32":
                return DataType.FLOAT32;
            case "FP64":
                return DataType.FLOAT64;
            default:
                throw new IllegalArgumentException("Invalid KServe data type: " + type);
        }
    }

    static String toKServeDataType(DataType type) {
        switch (type) {
            case BOOLEAN:
                return "BOOL";
            case UINT8:
                return "UINT8";
            case INT8:
                return "INT8";
            case INT32:
                return "INT32";
            case INT64:
                return "INT64";
            case FLOAT16:
                return "FP16";
            case FLOAT32:
                return "FP32";
            case FLOAT64:
                return "FP64";
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

    static KServeTensor fromTensor(NDArray array, String name) {
        KServeTensor tensor = new KServeTensor();
        tensor.name = name;
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
        for (double d : data) {
            bb.put((byte) d);
        }
        bb.rewind();
        return bb;
    }
}
