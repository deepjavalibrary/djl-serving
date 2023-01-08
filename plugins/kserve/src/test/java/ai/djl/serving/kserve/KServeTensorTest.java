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

import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

public class KServeTensorTest {

    @Test
    public void testKServeTensor() {
        Assert.assertEquals(KServeTensor.fromKServeDataType("BOOL"), DataType.BOOLEAN);
        Assert.assertEquals(KServeTensor.fromKServeDataType("UINT8"), DataType.UINT8);
        Assert.assertEquals(KServeTensor.fromKServeDataType("INT8"), DataType.INT8);
        Assert.assertEquals(KServeTensor.fromKServeDataType("INT32"), DataType.INT32);
        Assert.assertEquals(KServeTensor.fromKServeDataType("INT64"), DataType.INT64);
        Assert.assertEquals(KServeTensor.fromKServeDataType("FP16"), DataType.FLOAT16);
        Assert.assertEquals(KServeTensor.fromKServeDataType("FP32"), DataType.FLOAT32);
        Assert.assertEquals(KServeTensor.fromKServeDataType("FP64"), DataType.FLOAT64);
        Assert.assertThrows(() -> KServeTensor.fromKServeDataType("UINT64"));

        Assert.assertEquals(KServeTensor.toKServeDataType(DataType.STRING), "string");

        Shape shape = new Shape(1);
        try (NDManager manager = NDManager.newBaseManager()) {
            for (DataType type : DataType.values()) {
                if (type != DataType.STRING
                        && type != DataType.UNKNOWN
                        && type != DataType.COMPLEX64) {
                    KServeTensor tensor = getKServeTensor(shape, type);
                    tensor.toTensor(manager);
                }
            }
        }
    }

    public static KServeTensor getKServeTensor(Shape shape, DataType dataType) {
        KServeTensor tensor = new KServeTensor();
        tensor.name = "input0";
        tensor.dataType = KServeTensor.toKServeDataType(dataType);
        tensor.shape = shape.getShape();
        tensor.data = new double[(int) shape.size() * dataType.getNumOfBytes()];
        return tensor;
    }
}
