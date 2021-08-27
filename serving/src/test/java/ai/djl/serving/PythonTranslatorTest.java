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

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Classifications;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.modality.cv.util.NDImageUtils;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.translate.TranslateException;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.security.GeneralSecurityException;
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
    private ModelServer server;
    private Field useNativeIoField;
    private Object previousUseNativeIoValue = true; // default value of useNativeIo is true.

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

            Path libsDir = modelDir.resolve("libs");
            Files.createDirectories(libsDir);

            libsDir = modelDir.resolve("bin");
            Files.createDirectories(libsDir);

            Path preProcessDestFile = modelDir.resolve("bin/pre_processing.py");
            Path preProcessSrcFile = Paths.get("../serving/src/test/resources/pre_processing.py");
            try (InputStream is = Files.newInputStream(preProcessSrcFile)) {
                Files.copy(is, preProcessDestFile, StandardCopyOption.REPLACE_EXISTING);
            }

            Path postProcessDestFile = modelDir.resolve("bin/post_processing.py");
            Path postProcessSrcFile = Paths.get("../serving/src/test/resources/post_processing.py");
            try (InputStream is = Files.newInputStream(postProcessSrcFile)) {
                Files.copy(is, postProcessDestFile, StandardCopyOption.REPLACE_EXISTING);
            }
        }

        Path imageFile = Paths.get("../serving/src/test/resources/0.png");
        try (InputStream is = Files.newInputStream(imageFile)) {
            data = Utils.toByteArray(is);
        }
    }

    @AfterClass
    public void tearDown() {
        Utils.deleteQuietly(modelDir);
    }

    @Test
    public void testImageClassificationUDS()
            throws NoSuchFieldException, IllegalAccessException, ModelException, TranslateException,
                    IOException, GeneralSecurityException, InterruptedException, ParseException {

        setUseNativeIoField(true);

        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "true");
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "pythonPath", "python3");
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "noOfPythonWorkers", "1");
        ConfigManagerTest.setConfiguration(
                ConfigManager.getInstance(), "startPythonWorker", "True");

        startModelServer();
        runPythonTranslator();
        stopModelServer();
        resetUseNativeIoField();
    }

    @Test
    public void testImageClassificationTCP()
            throws ModelException, NoSuchFieldException, IllegalAccessException, IOException,
                    TranslateException, GeneralSecurityException, InterruptedException,
                    ParseException {

        setUseNativeIoField(false);
        ConfigManager.init(ConfigManagerTest.parseArguments(new String[0]));
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "use_native_io", "false");
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "pythonPath", "python3");
        ConfigManagerTest.setConfiguration(ConfigManager.getInstance(), "noOfPythonWorkers", "5");
        ConfigManagerTest.setConfiguration(
                ConfigManager.getInstance(), "startPythonWorker", "True");
        startModelServer();
        runPythonTranslator();
        stopModelServer();
        resetUseNativeIoField();
    }

    private void setUseNativeIoField(boolean value) {
        try {
            if (ConfigManager.getInstance() != null) {
                useNativeIoField = Connector.class.getDeclaredField("useNativeIo");
                useNativeIoField.setAccessible(true);
                previousUseNativeIoValue = useNativeIoField.get(null);
                useNativeIoField.set(null, value);
            }
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    private void resetUseNativeIoField() {
        try {
            if (previousUseNativeIoValue != null && useNativeIoField != null) {
                useNativeIoField.set(null, previousUseNativeIoValue);
            }
        } catch (ReflectiveOperationException e) {
            throw new AssertionError(e);
        }
    }

    private void startModelServer()
            throws GeneralSecurityException, IOException, InterruptedException {
        server = new ModelServer(ConfigManager.getInstance());
        server.start();
    }

    private void stopModelServer() {
        server.stop();
    }

    private void runPythonTranslator() throws ModelException, IOException, TranslateException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(modelDir)
                        .optArgument("translator", "ai.djl.serving.pyclient.PythonTranslator")
                        .optArgument("preProcessor", "pre_processing.py:preprocess")
                        .optArgument("postProcessor", "post_processing.py:postprocess")
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
            Assert.assertEquals(output.getRequestId(), "1");

            // manually post process
            list = NDList.decode(manager, output.getContent());
            NDArray probabilities = list.singletonOrThrow().get(0).softmax(0);

            List<String> classes = model.getArtifact("synset.txt", Utils::readLines);
            Classifications result = new Classifications(classes, probabilities);
            logger.info("Classification result is " + JsonUtils.GSON.toJson(result));

            Assert.assertEquals(result.best().getClassName(), "0");
        }
    }
}
