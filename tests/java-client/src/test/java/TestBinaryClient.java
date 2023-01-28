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
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class TestBinaryClient {

    @Test
    public void testSendBinaryClient() throws IOException, InterruptedException {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray arr = manager.ones(new Shape(1, 3, 224, 224));
            NDList list = new NDList(arr);
            NDList list2 = BinaryClient.postNDList(list);
            Assert.assertEquals(list2.get(0).toFloatArray().length, 1000);
        }
    }
}
