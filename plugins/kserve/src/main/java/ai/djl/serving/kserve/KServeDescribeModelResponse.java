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

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.List;

/** This class represents the response for KServe model metadata request. */
public class KServeDescribeModelResponse {

    String name;
    List<String> versions;
    String platform;
    List<KServeTensor> inputs;
    List<KServeTensor> outputs;

    /** Constructs a {@code DescribeKServeModelResponse} instance. */
    public KServeDescribeModelResponse() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
    }

    /**
     * Returns the name of the model.
     *
     * @return the model name
     */
    public String getName() {
        return name;
    }

    /**
     * Returns the string list of model versions.
     *
     * @return list of model versions.
     */
    public List<String> getVersions() {
        return versions;
    }

    /**
     * Returns the engine of the model.
     *
     * @return the engine name.
     */
    public String getPlatform() {
        return platform;
    }

    void setPlatform(String engineName) {
        switch (engineName) {
            case "TensorRT":
                platform = "tensorrt_plan";
                break;
            case "MXNet":
                platform = "mxnet_mxnet";
                break;
            case "PyTorch":
                platform = "pytorch_torchscript";
                break;
            case "TensorFlow":
                platform = "tensorflow_savedmodel";
                break;
            case "OnnxRuntime":
                platform = "onnx_onnxv1";
                break;
            default:
                platform = engineName;
                break;
        }
    }

    void addInput(String name, DataType dataType, Shape shape) {
        inputs.add(new KServeTensor(name, shape.getShape(), dataType));
    }

    void addOutput(String name, DataType dataType, Shape shape) {
        outputs.add(new KServeTensor(name, shape.getShape(), dataType));
    }
}
