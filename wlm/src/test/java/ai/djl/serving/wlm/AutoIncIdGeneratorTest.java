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
package ai.djl.serving.wlm;

import ai.djl.serving.wlm.util.AutoIncIdGenerator;

import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Testing the {@link AutoIncIdGenerator}.
 *
 * @author erik.bamberg@web.de
 */
public class AutoIncIdGeneratorTest {

    @Test
    public void testGeneration() {
        AutoIncIdGenerator generator = new AutoIncIdGenerator("pref");
        Assert.assertNotEquals(generator.generate(), generator.generate());
        Assert.assertTrue(generator.generate().startsWith("pref"));
        Assert.assertNotEquals(
                generator.stripPrefix(generator.generate()),
                generator.stripPrefix(generator.generate()));
    }
}
