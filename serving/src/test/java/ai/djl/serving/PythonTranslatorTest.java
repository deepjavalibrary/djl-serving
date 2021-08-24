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
import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.metric.Metrics;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.pyclient.PythonTranslator;
import ai.djl.serving.util.ConfigManager;
import ai.djl.translate.TranslateException;
import ai.djl.translate.TranslatorContext;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.List;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/** This class contains tests for Python translator. */
public class PythonTranslatorTest {

    private static final Logger logger = LoggerFactory.getLogger(PythonTranslatorTest.class);

    private Path modelDir = Paths.get("build/models/mlp");
    private byte[] data;

    @BeforeClass
    public void setup() throws ModelException, IOException, ParseException {
        Utils.deleteQuietly(modelDir);
        Files.createDirectories(modelDir);
        Criteria<Image, Classifications> criteria =
                Criteria.builder()
                        .setTypes(Image.class, Classifications.class)
                        .optModelUrls("djl://ai.djl.mxnet/mlp/0.0.1/mlp")
                        .build();

        try (ZooModel<Image, Classifications> model = criteria.loadModel()) {
            Path symbolFile = modelDir.resolve("mlp-symbol.json");
            try (InputStream is = model.getArtifactAsStream("mlp-symbol.json")) {
                Files.copy(is, symbolFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path synsetFile = modelDir.resolve("synset.txt");
            try (InputStream is = model.getArtifactAsStream("synset.txt")) {
                Files.copy(is, synsetFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path paramFile = modelDir.resolve("mlp-0000.params");
            try (InputStream is = model.getArtifactAsStream("mlp-0000.params")) {
                Files.copy(is, paramFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path libsDir = modelDir.resolve("bin");
            Files.createDirectories(libsDir);

            Path preProcessFile = modelDir.resolve("bin/pre_processing.py");
            try (InputStream is = model.getArtifactAsStream("pre_processing.py")) {
                Files.copy(is, preProcessFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        Path imageFile = Paths.get("../serving/src/test/resources/0.png");
        try (InputStream is = Files.newInputStream(imageFile)) {
            data = Utils.toByteArray(is);
        }

        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));
    }

    @AfterClass
    public void tearDown() {
        Utils.deleteQuietly(modelDir);
    }

    @Test(enabled = false)
    public void testImageClassification()
            throws ModelException, NoSuchFieldException, IllegalAccessException, IOException,
                    TranslateException {
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "false");
        runPythonTranslator();
    }

    @Test(enabled = false)
    public void testPythonTranslatorTCP()
            throws IOException, TranslateException, NoSuchFieldException, IllegalAccessException {
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "false");
        testPythonTranslator();
    }

    @Test(enabled = false)
    public void testPythonTranslatorUDS()
            throws IOException, TranslateException, NoSuchFieldException, IllegalAccessException {
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "true");
        testPythonTranslator();
    }

    private void runPythonTranslator() throws ModelException, IOException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optArgument("translator", "ai.djl.serving.pyclient.PythonTranslator")
                        .optArgument("preProcessor", "pre_processing.py:preprocess")
                        .build();

        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            NDManager manager = model.getNDManager();

            // manually pre process
            ByteArrayInputStream is = new ByteArrayInputStream(data);
            Image image = ImageFactory.getInstance().fromInputStream(is);
            NDArray array = image.toNDArray(manager, Image.Flag.GRAYSCALE);
            array = NDImageUtils.toTensor(array).expandDims(0);
            NDList list = new NDList(array);

            Input input = new Input("1");
            input.addData(null, list.encode());
            Output output = predictor.predict(input);
            // Assert.assertEquals(output.getRequestId(), "1");

            // manually post process
            list = NDList.decode(manager, output.getContent());
            NDArray probabilities = list.singletonOrThrow().get(0).softmax(0);

            List<String> classes = model.getArtifact("synset.txt", Utils::readLines);
            Classifications result = new Classifications(classes, probabilities);
            logger.info("Classification result is " + JsonUtils.GSON.toJson(result));

            Assert.assertEquals(result.best().getClassName(), "0");
        }
    }

    private void testPythonTranslator() throws TranslateException, IOException {
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
