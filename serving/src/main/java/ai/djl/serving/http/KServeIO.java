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

import java.util.List;

/**
 * This class represents the KServe inputs and output structure for KServe infer and model request.
 */
public class KServeIO {
    private final String name;
    private final String dataType;
    private final List<Long> shape;
    private List<Double> data;

    /**
     * Constructs a {@code KServeIO} instance.
     *
     * @param name the IO name
     * @param dataType the IO datatype
     * @param shape the IO id
     * @param data the IO data
     */
    public KServeIO(String name, String dataType, List<Long> shape, List<Double> data) {
        this.name = name;
        this.dataType = dataType;
        this.shape = shape;
        this.data = data;
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

    /**
     * Returns the data of the Input.
     *
     * @return input data.
     */
    public List<Double> getData() {
        return data;
    }

    /**
     * Sets the IO data.
     *
     * @param data the IO data
     */
    public void setData(List<Double> data) {
        this.data = data;
    }

    /** {@inheritDoc} */
    @Override
    public String toString() {
        return "Input [ name: "
                + name
                + ", dataType: "
                + dataType
                + ", shape: "
                + shape
                + ", data "
                + data
                + " ]";
    }
}
