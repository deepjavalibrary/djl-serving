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
import ai.djl.inference.streaming.ChunkedBytesSupplier;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.BytesSupplier;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.TranslateException;
import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.stream.Stream;

/** Tests for the session manager. */
public class PySessionsTests {

    @Test
    public void testLocalLoadSave()
            throws TranslateException, IOException, ModelException, InterruptedException {
        Path path = Paths.get("build/djl_sessions");
        Utils.deleteQuietly(path);

        Criteria<Input, Output> criteria =
                Criteria.builder()
                        .setTypes(Input.class, Output.class)
                        .optModelPath(Paths.get("src/test/resources/stateful"))
                        .optEngine("Python")
                        .optOption("sessions_path", path.toAbsolutePath().toString())
                        .optOption("sessions_expiration", "1")
                        .build();
        try (ZooModel<Input, Output> model = criteria.loadModel();
                Predictor<Input, Output> predictor = model.newPredictor()) {
            // test create session
            Input createSession = new Input();
            createSession.addProperty("Content-Type", "application/json");
            createSession.add(BytesSupplier.wrapAsJson(Map.of("action", "create_session")));
            Output ret = predictor.predict(createSession);
            String sessionId = ret.getProperty("X-Amzn-SageMaker-Session-Id", null);
            Assert.assertNotNull(sessionId);
            Assert.assertEquals(ret.getProperty("Content-Type", null), "application/json");

            // test regular request
            Input regular = new Input();
            regular.addProperty("Content-Type", "application/json");
            regular.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
            regular.add(BytesSupplier.wrapAsJson(Map.of("action", "regular")));
            ret = predictor.predict(regular);
            Assert.assertEquals(ret.getProperty("Content-Type", null), "application/json");
            Assert.assertTrue(ret.getAsString(0).contains("(10, 5, 5)"));

            // test streaming request
            Input stream = new Input();
            stream.addProperty("Content-Type", "application/json");
            stream.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
            stream.add(BytesSupplier.wrapAsJson(Map.of("action", "streaming")));
            ret = predictor.predict(stream);
            Assert.assertEquals(ret.getProperty("Content-Type", null), "application/jsonlines");
            BytesSupplier data = ret.getData();
            Assert.assertTrue(data instanceof ChunkedBytesSupplier);
            String content = data.getAsString();
            Assert.assertEquals(content.split("\n").length, 10);

            // test session timeout
            Thread.sleep(1000);
            ret = predictor.predict(createSession);
            sessionId = ret.getProperty("X-Amzn-SageMaker-Session-Id", null);
            Assert.assertNotNull(sessionId);
            long count;
            try (Stream<Path> files = Files.list(path)) {
                count = files.count();
            }
            Assert.assertEquals(count, 1);

            // test close session with missing sessionId
            Input closeSession = new Input();
            closeSession.addProperty("Content-Type", "application/json");
            closeSession.add(BytesSupplier.wrapAsJson(Map.of("action", "close_session")));
            ret = predictor.predict(closeSession);
            Assert.assertEquals(ret.getProperty("Content-Type", null), "application/json");
            Assert.assertTrue(ret.getAsString(0).contains("invalid session_id"));
            try (Stream<Path> files = Files.list(path)) {
                count = files.count();
            }
            Assert.assertEquals(count, 1);

            // test close session
            closeSession.addProperty("X-Amzn-SageMaker-Session-Id", sessionId);
            ret = predictor.predict(closeSession);
            Assert.assertEquals(ret.getProperty("X-Amzn-SageMaker-Session-Closed", null), "true");
            Assert.assertEquals(ret.getProperty("Content-Type", null), "application/json");
            Assert.assertTrue(ret.getAsString(0).contains("session closed"));
            try (Stream<Path> files = Files.list(path)) {
                count = files.count();
            }
            Assert.assertEquals(count, 0);
        }
    }
}
