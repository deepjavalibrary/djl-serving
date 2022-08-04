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

import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/** This class represents the response for KServe model metadata request. */
public class KServeDescribeModelResponse {
    private String name;
    private List<String> versions;
    private String platform;
    private List<KServeIO> inputs;
    private List<KServeIO> outputs;

    /** Constructs a {@code DescribeKServeModelResponse} instance. */
    public KServeDescribeModelResponse() {
        versions = new ArrayList<>();
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();
    }

    /**
     * Sets the model name.
     *
     * @param name the model name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Sets the platform.
     *
     * @param platform the platform.
     */
    public void setPlatform(String platform) {
        this.platform = platform;
    }

    /**
     * Sets the platform given the engine name.
     *
     * @param engineName Name of the engine.
     */
    public void setPlatformForEngineName(String engineName) {
        this.platform = Platform.getKServePlatformForEngine(engineName);
    }

    /**
     * Sets the list of a model's versions.
     *
     * @param versions versions of a model.
     */
    public void setVersions(List<String> versions) {
        this.versions = versions;
    }

    /**
     * Adds the input to the list of inputs.
     *
     * @param name name of the input.
     * @param dataType datatype of the input.
     * @param shape shape of the input.
     */
    public void addInput(String name, DataType dataType, Shape shape) {
        List<Long> shapeList = Arrays.stream(shape.getShape()).boxed().collect(Collectors.toList());
        KServeDataType kdt = KServeDataType.getKServeDtForNDArrayDT(dataType);
        this.inputs.add(new KServeIO(name, kdt.name(), shapeList));
    }

    /**
     * Adds the output to the list of outputs.
     *
     * @param name name of the output.
     * @param dataType datatype of the output.
     * @param shape shape of the output.
     */
    public void addOutput(String name, DataType dataType, Shape shape) {
        List<Long> shapeList = Arrays.stream(shape.getShape()).boxed().collect(Collectors.toList());
        KServeDataType kdt = KServeDataType.getKServeDtForNDArrayDT(dataType);
        this.outputs.add(new KServeIO(name, kdt.name(), shapeList));
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

    /**
     * Returns the input list of the model.
     *
     * @return the input list
     */
    public List<KServeIO> getInputs() {
        return inputs;
    }

    /**
     * Returns the output list of the model.
     *
     * @return the output list
     */
    public List<KServeIO> getOutputs() {
        return outputs;
    }

    /**
     * This class represents the definition of inputs required and output generated by the model.
     */
    private static final class KServeIO {
        private String name;
        private String dataType;
        private List<Long> shape;

        public KServeIO(String name, String dataType, List<Long> shape) {
            this.name = name;
            this.dataType = dataType;
            this.shape = shape;
        }

        /**
         * Returns the name of the Input.
         *
         * @return input name
         */
        public String getName() {
            return name;
        }

        /**
         * Returns the datatype of the Input.
         *
         * @return input datatype
         */
        public String getDataType() {
            return dataType;
        }

        /**
         * Returns the shape of the Input.
         *
         * @return input shape.
         */
        public List<Long> getShape() {
            return shape;
        }
    }

    private enum Platform {
        TRT_PLAN("plan", "TensorRT"),
        TF_SAVEDMODEL("savedmodel", "TensorFlow"),
        ONNXV1("onnxv1", "OnnxRuntime"),
        TORCHSCRIPT("torchscript", "PyTorch"),
        MXNET("mxnet", "MXNet");

        private String modelType; // model_type supported by engine.
        private String engineName; // engine name in DJL format.

        Platform(String modelType, String engineName) {
            this.modelType = modelType;
            this.engineName = engineName;
        }

        /**
         * Returns the platform name for the respective engine. In KServe protocol, platform name is
         * engine name and the model type supported by the engine, separated by underscore.
         *
         * @param engineName name of the engine.
         * @return Plaform name in KServe format.
         */
        public static String getKServePlatformForEngine(String engineName) {
            for (Platform platform : Platform.values()) {
                if (platform.engineName.equals(engineName)) {
                    return platform.engineName + "_" + platform.modelType;
                }
            }
            return engineName;
        }
    }

    /** This class represents the Kserve Datatype of the inputs and outputs of the model. */
    private enum KServeDataType {
        FP32(DataType.FLOAT32),
        FP64(DataType.FLOAT64),
        FP16(DataType.FLOAT16),
        UINT8(DataType.UINT8),
        INT32(DataType.INT32),
        INT8(DataType.INT8),
        INT64(DataType.INT64),
        BOOL(DataType.BOOLEAN),
        BYTES(DataType.UNKNOWN);

        private DataType dataType;

        KServeDataType(DataType dataType) {
            this.dataType = dataType;
        }

        /**
         * Returns the corresponding KServe datatype for the given NDArray datatype.
         *
         * @param dataType NDArray datatype
         * @return KServe datatype
         */
        public static KServeDataType getKServeDtForNDArrayDT(DataType dataType) {
            for (KServeDataType kdt : KServeDataType.values()) {
                if (kdt.dataType.name().equals(dataType.name())) {
                    return kdt;
                }
            }

            return BYTES;
        }
    }
}
