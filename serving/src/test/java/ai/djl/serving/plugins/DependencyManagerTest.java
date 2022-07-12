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
package ai.djl.serving.plugins;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;

public class DependencyManagerTest {

    @Test
    public void testInstallDependency() throws IOException {
        DependencyManager dm = DependencyManager.getInstance();
        dm.installEngine("OnnxRuntime");
        dm.installDependency("ai.djl.pytorch:pytorch-jni:1.11.0-0.18.0");

        Assert.assertThrows(() -> dm.installDependency("ai.djl.pytorch:pytorch-jni"));
    }
}
