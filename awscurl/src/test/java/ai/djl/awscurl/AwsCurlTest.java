/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

package ai.djl.awscurl;

import ai.djl.util.Utils;

import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.util.AsciiString;

import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterTest;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class AwsCurlTest {

    private TestHttpServer server;

    @BeforeClass
    public void setUp() throws Exception {
        server = new TestHttpServer();
        server.start();
    }

    @AfterClass
    public void tierDown() {
        server.stop();
    }

    @AfterTest
    public void cleanUp() {
        System.clearProperty("TOKENIZER");
        System.clearProperty("EXCLUDE_INPUT_TOKEN");
        System.clearProperty(AwsCredentials.ACCESS_KEY_ENV_VAR);
        System.clearProperty(AwsCredentials.SECRET_KEY_ENV_VAR);
        System.clearProperty("AWS_REGION");
    }

    @Test
    public void testHelp() {
        String[] args = {"--help"};
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testOutput() throws IOException {
        TestHttpHandler.setContent(
                "{\"generated_text\": [\"test out\"]}", HttpHeaderValues.TEXT_PLAIN);
        Path out = Paths.get("build/output");
        Utils.deleteQuietly(out);
        Files.createDirectories(out);
        String[] args = {
            "http://localhost:18080/invocations",
            "-G",
            "-d",
            "model=test",
            "-o",
            "build/output/output"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
        Assert.assertTrue(Files.exists(out.resolve("output.0")));
    }

    @Test
    public void testDelay() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {"http://localhost:18080/invocations", "--delay", "5"};
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());

        args = new String[] {"http://localhost:18080/invocations", "--delay", "rand(1, 5)"};
        ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());

        args = new String[] {"http://localhost:18080/invocations", "--delay", "rand(5)"};
        ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testDataset() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations", "--dataset", "src/test/resources/prompts"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());

        args =
                new String[] {
                    "http://localhost:18080/invocations",
                    "--dataset",
                    "src/test/resources/prompts/prompt1.txt"
                };
        ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testFormData() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations",
            "-F",
            "data=@src/test/resources/prompts/prompt1.txt",
            "-F",
            "data=@src/test/resources/prompts/prompt1.txt;type=text/plain",
            "--form-string",
            "param=test;type=application/json;filename=a.json",
            "--form-string",
            "param=@test;filename=a.json"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());

        args =
                new String[] {
                    "http://localhost:18080/invocations",
                    "--form-string",
                    "filename=a.jpg",
                    "--form-string",
                    "data=;type=application/json;filename=b.json",
                    "--form-string",
                    "data=1;filename=b.json"
                };
        ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testFileUpload() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations",
            "-T",
            "src/test/resources/prompts/prompt1.txt",
            "-H",
            "User-Agent: awscurl"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testUrlEncodedData() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations",
            "-d",
            "@src/test/resources/prompts/prompt1.txt",
            "--data-urlencode",
            "data=1&=value&@src/test/resources/prompts/prompt1.txt",
            "--data-raw",
            "@src/test/resources/prompts/prompt1.txt"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testPlainText() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations?",
            "--connect-timeout",
            "60",
            "-i",
            "-k",
            "-H",
            "Content-type: application/json",
            "-G",
            "-d",
            "model=test",
            "-N",
            "2"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertNull(ret.getTokenThroughput());
    }

    @Test
    public void testPlainTextTokenCount() {
        TestHttpHandler.setContent("Hello world", HttpHeaderValues.TEXT_PLAIN);
        String[] args = {
            "http://localhost:18080/invocations?param=1",
            "-v",
            "-H",
            "Content-type: application/json",
            "-G",
            "-d",
            "model=test",
            "-c",
            "1",
            "-N",
            "1",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertNotNull(ret.getTokenThroughput());
        Assert.assertEquals(ret.getTotalTokens(), 2);
    }

    @Test
    public void testJson() {
        System.clearProperty("TOKENIZER");
        TokenUtils.setTokenizer(); // reset tokenizer
        // single outputs
        TestHttpHandler.setContent(
                "{\"generated_text\": \"Hello world\"}", HttpHeaderValues.APPLICATION_JSON);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-c",
            "1",
            "-N",
            "2",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 4);
        Assert.assertNotNull(ret.getTokenThroughput());

        System.setProperty("TOKENIZER", "gpt2");
        TokenUtils.setTokenizer();

        // multiple outputs
        TestHttpHandler.setContent(
                "{\"generated_text\": [\"Hello world\", \"Bonjour\"]}",
                HttpHeaderValues.APPLICATION_JSON);
        ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 10);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    @Test
    public void testLegacyJsonLines() {
        System.setProperty("TOKENIZER", "gpt2");
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString contentType = AsciiString.cached("application/jsonlines");
        TestHttpHandler.setContent(
                "{\"outputs\": [\"Hello\"]}\n{\"outputs\": [\" world\"]}", contentType);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-c",
            "1",
            "-N",
            "2",
            "-P",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 4);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    @Test
    public void testJsonLinesWithTokenDetails() {
        System.setProperty("TOKENIZER", "gpt2");
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString contentType = AsciiString.cached("application/jsonlines");
        TestHttpHandler.setContent(
                "{\"token\": {}}\n"
                        + "{\"token\": {}}\n"
                        + "{\"token\": {}, \"generated_text\"=\"Hello world.\", \"details\": {}}",
                contentType);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-c",
            "1",
            "-N",
            "2",
            "-P",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 6);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    @Test
    public void testServerSentEvent() {
        System.setProperty("TOKENIZER", "gpt2");
        System.setProperty("EXCLUDE_INPUT_TOKEN", "true");
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString contentType = AsciiString.cached("text/event-stream");
        TestHttpHandler.setContent(
                "data: {\"token\": {}}\n\n"
                        + "data: {\"token\": {}}\n\n"
                        + "data: {\"token\": {}, \"generated_text\"=\"prompt Hello world.\","
                        + " \"details\": {}}",
                contentType);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{\"inputs\":[\"prompt\"]}",
            "-c",
            "1",
            "-N",
            "2",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 6);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    @Test
    public void testCustomJsonKey() {
        System.setProperty("TOKENIZER", "gpt2");
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString contentType = AsciiString.cached("text/event-stream");
        TestHttpHandler.setContent(
                "data: {\"token\": {\"text\": \"Hello\"}}\n\n"
                        + "data: {\"token\": {\"text\": \" world\"}}\n\n"
                        + "data: {\"token\": {}, \"generated_text\"=\"Hello world.\","
                        + " \"details\": {}}",
                contentType);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{\"inputs\":[\"Hello workd\"]}",
            "-c",
            "1",
            "-N",
            "2",
            "-j",
            "token/text",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 4);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    @Test
    public void testCoralStream() {
        System.setProperty("TOKENIZER", "gpt2");
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString contentType = AsciiString.cached("application/vnd.amazon.eventstream");
        byte[] line1 = buildCoralEvent("{\"outputs\": [\"Hello\"]}\r\n");
        byte[] line2 = buildCoralEvent("{\"outputs\": [\" world\"]}\r\n");
        byte[] content = new byte[line1.length + line2.length];
        System.arraycopy(line1, 0, content, 0, line1.length);
        System.arraycopy(line2, 0, content, line1.length, line2.length);

        TestHttpHandler.setContent(content, contentType);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-c",
            "1",
            "-N",
            "2",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getTotalTokens(), 4);
        Assert.assertNotNull(ret.getTokenThroughput());
    }

    private byte[] buildCoralEvent(String payload) {
        byte[] data = payload.getBytes(StandardCharsets.UTF_8);
        int totalLength = data.length + 16;
        byte[] buf = new byte[totalLength];
        ByteBuffer bb = ByteBuffer.wrap(buf);
        bb.order(ByteOrder.BIG_ENDIAN);
        bb.putInt(totalLength);
        bb.putInt(0); // header length
        bb.putInt(0); // unknown
        bb.put(data);
        return buf;
    }

    @Test
    public void testSigner() {
        String[] args = {
            "-X",
            "POST",
            "-n",
            "sagemaker",
            "http://localhost:18080/invocations?model=test",
            "-d",
            "input"
        };
        System.setProperty(AwsCredentials.ACCESS_KEY_ENV_VAR, "id");
        System.setProperty(AwsCredentials.SECRET_KEY_ENV_VAR, "key");
        System.setProperty("AWS_REGION", "us-east-1");
        Result ret = AwsCurl.run(args);
        Assert.assertFalse(ret.hasError());
    }

    @Test
    public void testInvalidCmd() throws IOException {
        // missing URL
        Result ret = AwsCurl.run(new String[] {"-v"});
        Assert.assertTrue(ret.hasError());

        ret = AwsCurl.run(new String[] {"--connect-timeout", "a", "http://localhost"});
        Assert.assertTrue(ret.hasError());

        ret = AwsCurl.run(new String[] {"-c", "0", "http://localhost"});
        Assert.assertTrue(ret.hasError());

        ret = AwsCurl.run(new String[] {"-c", "a", "http://localhost"});
        Assert.assertTrue(ret.hasError());

        ret = AwsCurl.run(new String[] {"-N", "-1", "http://localhost"});
        Assert.assertTrue(ret.hasError());

        ret = AwsCurl.run(new String[] {"-N", "a", "http://localhost"});
        Assert.assertTrue(ret.hasError());

        String[] args =
                new String[] {
                    "http://localhost:18080/invocations", "--dataset", "non-exist/prompts"
                };
        ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());

        Path path = Paths.get("build/empty");
        Files.createDirectories(path);
        args = new String[] {"http://localhost:18080/invocations", "--dataset", path.toString()};
        ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());
    }

    @Test
    public void testInvalidUrl() {
        String[] args = {"-v", "localhost:18080/invocations", "-d", "input"};
        Result ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());

        args = new String[] {"c:\\folder\\f", "-d", "input"};
        ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());
    }

    @Test
    public void testNoCredential() {
        String[] args = {"-n", "sagemaker", "http://localhost:18080/invocations", "-d", "input"};
        Result ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());

        // failed to inferRegion
        System.setProperty(AwsCredentials.ACCESS_KEY_ENV_VAR, "id");
        System.setProperty(AwsCredentials.SECRET_KEY_ENV_VAR, "key");
        ret = AwsCurl.run(args);
        Assert.assertTrue(ret.hasError());
    }

    @Test
    public void testInvalidJson() {
        TokenUtils.setTokenizer(); // reset tokenizer
        TestHttpHandler.setContent("{\"token\": \"Hello world}", HttpHeaderValues.APPLICATION_JSON);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-N",
            "2",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getFailedRequests(), 2);
    }

    @Test
    public void testInvalidJsonLines() {
        TokenUtils.setTokenizer(); // reset tokenizer
        AsciiString jsonlines = AsciiString.cached("application/jsonlines");
        TestHttpHandler.setContent("outputs: Hello", jsonlines);
        String[] args = {
            "http://localhost:18080/invocations",
            "-H",
            "Content-type: application/json",
            "-d",
            "{}",
            "-N",
            "2",
            "-t"
        };
        Result ret = AwsCurl.run(args);
        Assert.assertEquals(ret.getFailedRequests(), 2);
    }
}
