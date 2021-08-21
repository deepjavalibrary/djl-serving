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
package ai.djl.serving;

import ai.djl.Model;
import ai.djl.metric.Metrics;
import ai.djl.modality.Input;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.serving.pyclient.PythonTranslator;
import ai.djl.serving.util.ConfigManager;
import ai.djl.translate.TranslatorContext;
import org.apache.commons.cli.ParseException;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/** This class contains tests for Python translator. */
public class PythonTranslatorTest {

    @BeforeClass
    public void setup() throws ParseException {
        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));
    }

    @Test(enabled = false)
    public void testPythonTranslatorTCP() throws Exception {
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "false");
        testPythonTranslator();
    }

    @Test(enabled = false)
    public void testPythonTranslatorUDS() throws Exception {
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "true");
        testPythonTranslator();
    }

    private void testPythonTranslator() throws Exception {
        try (NDManager manager = NDManager.newBaseManager()) {
            NDArray ndArray = manager.zeros(new Shape(2, 2));
            NDList ndList = new NDList(ndArray);
            Input input = new Input("1");
            input.addData(ndList.encode());

            PythonTranslator pythonTranslator = new PythonTranslator();
            TranslatorContext context =
                    new TranslatorContext() {

                        @Override
                        public Model getModel() {
                            return null;
                        }

                        @Override
                        public NDManager getNDManager() {
                            return manager;
                        }

                        @Override
                        public Metrics getMetrics() {
                            return null;
                        }

                        @Override
                        public Object getAttachment(String key) {
                            return null;
                        }

                        @Override
                        public void setAttachment(String key, Object value) {}

                        @Override
                        public void close() {}
                    };

            NDList list = pythonTranslator.processInput(context, input);
            Assert.assertFalse(list.isEmpty());
        }
    }
}
