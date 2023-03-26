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

import ai.djl.ndarray.types.DataType;

/** Helper to convert between {@link DataType} an the TritonServer DataTypes. */
public enum TsDataType {
    INVALID,
    BOOL,
    UINT8,
    UINT16,
    UINT32,
    UINT64,
    INT8,
    INT16,
    INT32,
    INT64,
    FP16,
    FP32,
    FP64,
    BYTES,
    BF16;

    DataType toDataType() {
        switch (this) {
            case BOOL:
                return DataType.BOOLEAN;
            case UINT8:
                return DataType.UINT8;
            case INT8:
                return DataType.INT8;
            case INT32:
                return DataType.INT32;
            case INT64:
                return DataType.INT64;
            case FP16:
                return DataType.FLOAT16;
            case FP32:
                return DataType.FLOAT32;
            case FP64:
                return DataType.FLOAT64;
            case BYTES:
                return DataType.STRING;
            case UINT16:
            case UINT32:
            case UINT64:
            case INT16:
            case BF16:
            case INVALID:
            default:
                return DataType.UNKNOWN;
        }
    }
}
