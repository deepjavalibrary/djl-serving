/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.python.engine;

import ai.djl.ModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.RandomUtils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Tests for the session manager. */
public class PySessionsTests {

    @Test
    public void testLocalLoadSave() throws TranslateException, IOException, ModelException {
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "local");
        testLoadSave("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testLocalPrune() throws TranslateException, IOException, ModelException {
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "local");
        testPrune("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testFilesLoadSave() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "files");
        options.put("sessions_path", path.toAbsolutePath().toString());
        testLoadSave("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testFilesPersistence() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "files");
        options.put("sessions_path", path.toAbsolutePath().toString());
        testPersistence("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testFilesParallel() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "files");
        options.put("sessions_path", path.toAbsolutePath().toString());
        testParallel("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testFilesPrune() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "files");
        options.put("sessions_path", path.toAbsolutePath().toString());
        testPrune("src/test/resources/sessionecho/simple", options);
    }

    @Test
    public void testMmapLoadSave() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "mmap");
        options.put("sessions_path", path.toAbsolutePath().toString());
        options.put("sessions_file_size", "1");
        testLoadSave("src/test/resources/sessionecho/mmap", options);
    }

    @Test
    public void testMmapPersistence() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "mmap");
        options.put("sessions_path", path.toAbsolutePath().toString());
        options.put("sessions_file_size", "1");
        testPersistence("src/test/resources/sessionecho/mmap", options);
    }

    @Test
    public void testMmapParallel() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "mmap");
        options.put("sessions_path", path.toAbsolutePath().toString());
        options.put("sessions_file_size", "1");
        testParallel("src/test/resources/sessionecho/mmap", options);
    }

    @Test
    public void testMmapPrune() throws TranslateException, IOException, ModelException {
        Path path = resetDirectory(Paths.get("build/sessionTest"));
        Map<String, String> options = new ConcurrentHashMap<>();
        options.put("session_manager", "mmap");
        options.put("sessions_path", path.toAbsolutePath().toString());
        options.put("sessions_file_size", "1");
        testPrune("src/test/resources/sessionecho/mmap", options);
    }

    private void testLoadSave(String modelPath, Map<String, String> options)
            throws TranslateException, IOException, ModelException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get(modelPath))
                        .optEngine("Python")
                        .optOptions(options)
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            Map<String, Integer> sessionCounts = new ConcurrentHashMap<>(3);
            for (int i = 0; i < 20; i++) {

                // Choose a random session, and increment the session count
                String sessionId = Integer.toString(RandomUtils.nextInt(3));
                sessionCounts.compute(sessionId, (k, v) -> v != null ? v + 1 : 1);
                int sessionCount = sessionCounts.get(sessionId);

                // Run the sessionecho model with the session
                Input input = new Input();
                input.add("input");
                input.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
                Output output = predictor.predict(input);
                Assert.assertEquals(output.getData().getAsString(), sessionCount + "input");
            }
        }
    }

    private void testPersistence(String modelPath, Map<String, String> options)
            throws TranslateException, IOException, ModelException {
        for (int i = 1; i <= 3; i++) {
            Criteria<Input, Output> criteria =
                    Criteria.builder()
                            .setTypes(Input.class, Output.class)
                            .optModelPath(Paths.get(modelPath))
                            .optEngine("Python")
                            .optOptions(options)
                            .build();
            try (ZooModel<Input, Output> model = criteria.loadModel();
                    Predictor<Input, Output> predictor = model.newPredictor()) {

                // Run the sessionecho model with the session
                Input input = new Input();
                input.add("input");
                input.addProperty("X-Amzn-SageMaker-Session-Id", "sess");
                Output output = predictor.predict(input);
                Assert.assertEquals(output.getData().getAsString(), i + "input");
            }
        }
    }

    private void testParallel(String modelPath, Map<String, String> options)
            throws TranslateException, IOException, ModelException {
        List<ZooModel<Input, Output>> models = new ArrayList<>();
        List<Predictor<Input, Output>> predictors = new ArrayList<>();
        int numPredictors = 3;
        for (int i = 0; i < numPredictors; i++) {
            Criteria<Input, Output> criteria =
                    Criteria.builder()
                            .setTypes(Input.class, Output.class)
                            .optModelPath(Paths.get(modelPath))
                            .optEngine("Python")
                            .optOptions(options)
                            .build();
            ZooModel<Input, Output> model = criteria.loadModel();
            models.add(model);
            predictors.add(model.newPredictor());
        }
        Map<String, Integer> sessionCounts = new ConcurrentHashMap<>(3);
        for (int i = 0; i < 20; i++) {

            // Choose a random session, and increment the session count
            String sessionId = Integer.toString(RandomUtils.nextInt(3));
            sessionCounts.compute(sessionId, (k, v) -> v != null ? v + 1 : 1);
            int sessionCount = sessionCounts.get(sessionId);

            // Choose a random predictor
            Predictor<Input, Output> predictor = predictors.get(RandomUtils.nextInt(numPredictors));

            // Run the sessionecho model with the session
            Input input = new Input();
            input.add("input");
            input.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
            Output output = predictor.predict(input);
            Assert.assertEquals(output.getData().getAsString(), sessionCount + "input");
        }
        models.forEach(ZooModel::close);
        predictors.forEach(Predictor::close);
    }

    private void testPrune(String modelPath, Map<String, String> options)
            throws TranslateException, IOException, ModelException {
        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get(modelPath))
                        .optEngine("Python")
                        .optOptions(options)
                        .optOption("sessions_limit", "2")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            for (int i = 0; i < 4; i++) {

                // 3 sessions in cycle: 0, 1, 2, 0
                // Should prune 0 when calling 2 then need to re-init it when re-calling 0
                String sessionId = Integer.toString(i % 3);

                // Run the sessionecho model with the session
                Input input = new Input();
                input.add("input");
                input.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
                Output output = predictor.predict(input);
                Assert.assertEquals(output.getData().getAsString(), "1input");
            }
        }
    }

    private Path resetDirectory(Path path) throws IOException {
        File file = path.toFile();
        if (file.exists()) {
            Files.walk(path)
                    .sorted(Comparator.reverseOrder())
                    .map(Path::toFile)
                    .forEach(File::delete);
        }
        Assert.assertTrue(file.mkdirs());
        return path;
    }
}
