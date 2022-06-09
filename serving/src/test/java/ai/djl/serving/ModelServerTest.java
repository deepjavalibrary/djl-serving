/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotNull;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;

import ai.djl.modality.Classifications.Classification;
import ai.djl.serving.http.DescribeWorkflowResponse;
import ai.djl.serving.http.ErrorResponse;
import ai.djl.serving.http.ListModelsResponse;
import ai.djl.serving.http.ListWorkflowsResponse;
import ai.djl.serving.http.ListWorkflowsResponse.WorkflowItem;
import ai.djl.serving.http.ServerStartupException;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import ai.djl.util.cuda.CudaUtils;
import com.google.gson.reflect.TypeToken;
import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelHandler;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpContentDecompressor;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpHeaders;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpRequest;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpUtil;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder;
import io.netty.handler.codec.http.multipart.HttpPostRequestEncoder.ErrorDataEncoderException;
import io.netty.handler.codec.http.multipart.MemoryFileUpload;
import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.InsecureTrustManagerFactory;
import io.netty.handler.stream.ChunkedWriteHandler;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.internal.logging.InternalLoggerFactory;
import io.netty.util.internal.logging.Slf4JLoggerFactory;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStream;
import java.io.UnsupportedEncodingException;
import java.lang.reflect.Type;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

public class ModelServerTest {

    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private byte[] testImage;
    private List<EventLoopGroup> eventLoopGroups;
    volatile CountDownLatch latch;
    volatile HttpResponseStatus httpStatus;
    volatile String result;
    volatile HttpHeaders headers;

    static {
        try {
            SSLContext context = SSLContext.getInstance("TLS");
            context.init(null, InsecureTrustManagerFactory.INSTANCE.getTrustManagers(), null);

            HttpsURLConnection.setDefaultSSLSocketFactory(context.getSocketFactory());

            HttpsURLConnection.setDefaultHostnameVerifier((s, sslSession) -> true);
        } catch (GeneralSecurityException ignore) {
            // ignore
        }
    }

    @BeforeSuite
    public void beforeSuite() throws IOException {
        URL url = new URL("https://resources.djl.ai/images/0.png");
        try (InputStream is = url.openStream()) {
            testImage = Utils.toByteArray(is);
        }
        Path modelStore = Paths.get("build/models");
        Utils.deleteQuietly(modelStore);
        Files.createDirectories(modelStore);
        eventLoopGroups = new ArrayList<>();
        Path deps = Paths.get("deps");
        Files.createDirectories(deps);
        Path dest = deps.resolve("test.jar");
        ZipUtils.zip(Paths.get("build/classes/java/test/"), dest, true);
    }

    @AfterMethod
    public void afterMethod() {
        for (EventLoopGroup elg : eventLoopGroups) {
            elg.shutdownGracefully(0, 0, TimeUnit.SECONDS);
        }
        eventLoopGroups = new ArrayList<>();

        ModelManager.getInstance().clear();
    }

    @Test
    public void testModelStore()
            throws IOException, ServerStartupException, GeneralSecurityException, ParseException,
                    InterruptedException {
        try (ModelServer server = initTestServer("src/test/resources/config.properties")) {
            Path modelStore = Paths.get("build/models");
            Path modelDir = modelStore.resolve("test_model");
            Files.createDirectories(modelDir);
            Path notModel = modelStore.resolve("non-model");
            Files.createFile(notModel);

            String url = server.mapModelUrl(notModel); // not a model dir
            assertNull(url);

            url = server.mapModelUrl(modelDir); // empty folder
            assertNull(url);

            String expected = modelDir.toUri().toURL().toString();

            Path dlr = modelDir.resolve("test_model.so");
            Files.createFile(dlr);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::DLR:*=" + expected);

            Path xgb = modelDir.resolve("test_model.json");
            Files.createFile(xgb);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::XGBoost:*=" + expected);

            Path paddle = modelDir.resolve("__model__");
            Files.createFile(paddle);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::PaddlePaddle:*=" + expected);

            Path tflite = modelDir.resolve("test_model.tflite");
            Files.createFile(tflite);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::TFLite:*=" + expected);

            Path tensorRt = modelDir.resolve("test_model.uff");
            Files.createFile(tensorRt);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::TensorRT:*=" + expected);

            Path onnx = modelDir.resolve("test_model.onnx");
            Files.createFile(onnx);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::OnnxRuntime:*=" + expected);

            Path mxnet = modelDir.resolve("test_model-symbol.json");
            Files.createFile(mxnet);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::MXNet:*=" + expected);

            Path tensorflow = modelDir.resolve("saved_model.pb");
            Files.createFile(tensorflow);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::TensorFlow:*=" + expected);

            Path pytorch = modelDir.resolve("test_model.pt");
            Files.createFile(pytorch);
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::PyTorch:*=" + expected);

            Path prop = modelDir.resolve("serving.properties");
            try (BufferedWriter writer = Files.newBufferedWriter(prop)) {
                writer.write("engine=MyEngine");
            }
            url = server.mapModelUrl(modelDir);
            assertEquals(url, "test_model::MyEngine:*=" + expected);

            Path mar = modelStore.resolve("torchServe.mar");
            Path torchServe = modelStore.resolve("torchServe");
            Files.createDirectories(torchServe.resolve("MAR-INF"));
            ZipUtils.zip(torchServe, mar, false);

            url = server.mapModelUrl(mar);
            assertEquals(url, "torchServe::Python:*=" + mar.toUri().toURL());
        }
    }

    public static void main(String[] args)
            throws ReflectiveOperationException, ServerStartupException, GeneralSecurityException,
                    ErrorDataEncoderException, IOException, ParseException, InterruptedException {
        ModelServerTest t = new ModelServerTest();
        t.beforeSuite();
        t.test();
        t.afterMethod();
    }

    @Test
    public void test()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, ParseException, GeneralSecurityException,
                    ReflectiveOperationException, ServerStartupException {

        try (ModelServer server = initTestServer("src/test/resources/config.properties")) {
            assertTrue(server.isRunning());
            Channel channel = initTestChannel();

            assertNotNull(channel, "Failed to connect to inference port.");

            // inference API
            testRegisterModelTranslator(channel);

            testPing(channel);
            testRoot(channel);
            testPredictionsModels(channel);
            testInvocations(channel);
            testInvocationsMultipart(channel);
            testDescribeApi(channel);

            // management API
            testRegisterModel(channel);
            testRegisterModelAsync(channel);
            testRegisterWorkflow(channel);
            testRegisterWorkflowAsync(channel);
            testScaleModel(channel);
            testListModels(channel);
            testListWorkflows(channel);
            testDescribeModel(channel);
            testUnregisterModel(channel);

            testPredictionsInvalidRequestSize(channel);

            // plugin tests
            testStaticHtmlRequest();

            channel.close().sync();

            // negative test case that channel will be closed by server
            testInvalidUri();
            testInvalidPredictionsUri();
            testInvalidPredictionsMethod();
            testPredictionsModelNotFound();
            testInvalidDescribeModel();
            testDescribeModelNotFound();
            testInvalidManagementUri();
            testInvalidManagementMethod();
            testUnregisterModelNotFound();
            testInvalidScaleModel();
            testScaleModelNotFound();
            testRegisterModelMissingUrl();
            testRegisterModelNotFound();
            testRegisterModelConflict();
            testServiceUnavailable();

            ConfigManagerTest.testSsl();
        }
    }

    @Test
    public void testWorkflows()
            throws ServerStartupException, GeneralSecurityException, ParseException, IOException,
                    InterruptedException, ReflectiveOperationException {
        try (ModelServer server = initTestServer("src/test/resources/workflow.config.properties")) {
            assertTrue(server.isRunning());
            Channel channel = initTestChannel();

            testPredictionsModels(channel);
            testPredictionsWorkflows(channel);

            channel.close().sync();

            ConfigManagerTest.testSsl();
        }
    }

    private ModelServer initTestServer(String configFile)
            throws ParseException, ServerStartupException, GeneralSecurityException, IOException,
                    InterruptedException {
        String[] args = {"-f", configFile};
        Arguments arguments = ConfigManagerTest.parseArguments(args);
        assertFalse(arguments.hasHelp());

        ConfigManager.init(arguments);
        configManager = ConfigManager.getInstance();

        InternalLoggerFactory.setDefaultFactory(Slf4JLoggerFactory.INSTANCE);

        ModelServer server = new ModelServer(configManager);
        server.start();
        return server;
    }

    private Channel initTestChannel() throws InterruptedException {
        Channel channel = null;
        for (int i = 0; i < 5; ++i) {
            try {
                channel = connect(Connector.ConnectorType.MANAGEMENT);
                break;
            } catch (AssertionError e) {
                Thread.sleep(100);
            }
        }
        return channel;
    }

    private void testRoot(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        assertEquals(result, "{}\n");
    }

    private void testPing(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping");
        channel.writeAndFlush(req);
        latch.await();
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertNotNull(resp);
        assertTrue(headers.contains("x-request-id"));
    }

    private void testPredictionsModels(Channel channel) throws InterruptedException {
        String[] targets = new String[] {"/predictions/mlp"};
        testPredictions(channel, targets);
    }

    private void testPredictionsWorkflows(Channel channel) throws InterruptedException {
        String[] targets = new String[] {"/predictions/BasicWorkflow"};
        testPredictions(channel, targets);
    }

    private void testPredictions(Channel channel, String[] targets) throws InterruptedException {
        for (String target : targets) {
            reset();
            DefaultFullHttpRequest req =
                    new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, target);
            req.content().writeBytes(testImage);
            HttpUtil.setContentLength(req, req.content().readableBytes());
            req.headers()
                    .set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
            channel.writeAndFlush(req);

            latch.await();

            Type type = new TypeToken<List<Classification>>() {}.getType();
            List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
            assertEquals(classifications.get(0).getClassName(), "0");
        }
    }

    private void testInvocations(Channel channel) throws InterruptedException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);
        latch.await();

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testInvocationsMultipart(Channel channel)
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations?model_name=mlp");

        ByteBuf content = Unpooled.buffer(testImage.length);
        content.writeBytes(testImage);
        HttpPostRequestEncoder encoder = new HttpPostRequestEncoder(req, true);
        encoder.addBodyAttribute("test", "test");
        MemoryFileUpload body =
                new MemoryFileUpload("data", "0.png", "image/png", null, null, testImage.length);
        body.setContent(content);
        encoder.addBodyHttpData(body);

        channel.writeAndFlush(encoder.finalizeRequest());
        if (encoder.isChunked()) {
            channel.writeAndFlush(encoder).sync();
        }

        latch.await();

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testRegisterModelAsync(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();
        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_1&synchronous=false&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse statusResp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(statusResp.getStatus(), "Model \"mlp_1\" registration scheduled.");

        assertTrue(checkWorkflowRegistered(channel, "mlp_1"));
    }

    private void testRegisterModel(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();

        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"mlp_2\" registered.");
    }

    private void testRegisterModelTranslator(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();

        String url =
                "https://resources.djl.ai/demo/pytorch/traced_resnet18.zip?"
                        + "translator=ai.djl.serving.translator.TestTranslator"
                        + "&topK=1";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=res18&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"res18\" registered.");

        // send request
        reset();
        String imgUrl = "https://resources.djl.ai/images/kitten.jpg";
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/res18");
        req.content().writeBytes(imgUrl.getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        channel.writeAndFlush(req);
        latch.await();

        assertEquals(result, "topK: 1, best: n02124075 Egyptian cat");
    }

    private void testRegisterWorkflow(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();

        String url = "https://resources.djl.ai/test-models/basic-serving-workflow.json";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=" + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Workflow \"BasicWorkflow\" registered.");
    }

    private void testRegisterWorkflowAsync(Channel channel)
            throws InterruptedException, UnsupportedEncodingException {
        reset();
        String url = "https://resources.djl.ai/test-models/basic-serving-workflow2.json";
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?synchronous=false&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse statusResp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(statusResp.getStatus(), "Workflow \"BasicWorkflow2\" registration scheduled.");

        assertTrue(checkWorkflowRegistered(channel, "BasicWorkflow2"));
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=2&max_worker=4");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(
                resp.getStatus(),
                "Workflow \"mlp_2\" worker scaled. New Worker configuration min workers:2 max workers:4");
    }

    private void testListModels(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models");
        channel.writeAndFlush(req);
        latch.await();

        ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
        for (String expectedModel : new String[] {"mlp", "mlp_1", "mlp_2"}) {
            assertTrue(
                    resp.getModels()
                            .stream()
                            .anyMatch(w -> expectedModel.equals(w.getModelName())));
        }
    }

    private void testListWorkflows(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/workflows");
        channel.writeAndFlush(req);
        latch.await();

        ListWorkflowsResponse resp = JsonUtils.GSON.fromJson(result, ListWorkflowsResponse.class);
        for (String expectedWorkflow : new String[] {"mlp", "mlp_1", "mlp_2", "BasicWorkflow"}) {
            assertTrue(
                    resp.getWorkflows()
                            .stream()
                            .anyMatch(w -> expectedWorkflow.equals(w.getWorkflowName())));
        }
    }

    private void testDescribeModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/mlp_2");
        channel.writeAndFlush(req);
        latch.await();

        DescribeWorkflowResponse resp =
                JsonUtils.GSON.fromJson(result, DescribeWorkflowResponse.class);
        assertTrue(resp.getWorkers().size() > 1);

        assertEquals(resp.getWorkflowName(), "mlp_2");
        assertNotNull(resp.getWorkflowUrl());
        assertEquals(resp.getMinWorkers(), 2);
        assertEquals(resp.getMaxWorkers(), 4);
        assertEquals(resp.getBatchSize(), 1);
        assertEquals(resp.getMaxBatchDelay(), 100);
        assertEquals(resp.getStatus(), "Healthy");
        DescribeWorkflowResponse.Worker worker = resp.getWorkers().get(0);
        assertTrue(worker.getId() > 0);
        assertNotNull(worker.getStartTime());
        assertNotNull(worker.getStatus());
        assertEquals(worker.isGpu(), CudaUtils.hasCuda());
    }

    private void testUnregisterModel(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/mlp_1");
        channel.writeAndFlush(req);
        latch.await();

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model or workflow \"mlp_1\" unregistered");
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/mlp");
        channel.writeAndFlush(req);
        latch.await();

        assertEquals(result, "{}\n");
    }

    private void testStaticHtmlRequest() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        reset();
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/");
        channel.writeAndFlush(req).sync();
        latch.await();

        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req);

        latch.await();

        assertEquals(httpStatus, HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    private void testInvalidUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), "Model or workflow not found: InvalidModel");
        }
    }

    private void testInvalidPredictionsUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testPredictionsModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), "Model or workflow not found: InvalidModel");
        }
    }

    private void testInvalidManagementUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testInvalidManagementMethod() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
            assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
        }
    }

    private void testInvalidPredictionsMethod() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.METHOD_NOT_ALLOWED.code());
            assertEquals(resp.getMessage(), ERROR_METHOD_NOT_ALLOWED);
        }
    }

    private void testDescribeModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), "Workflow not found: InvalidModel");
        }
    }

    private void testRegisterModelMissingUrl() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
            assertEquals(resp.getMessage(), "Parameter url is required.");
        }
    }

    private void testRegisterModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/models?url=InvalidUrl");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(
                    resp.getMessage(), "No matching model with specified Input/Output type found.");
        }
    }

    private void testRegisterModelConflict()
            throws InterruptedException, UnsupportedEncodingException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        String url = "https://resources.djl.ai/test-models/mlp.tar.gz";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8.name()));
        channel.writeAndFlush(req);
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.BAD_REQUEST.code());
            assertEquals(resp.getMessage(), "Workflow mlp_2 is already registered.");
        }
    }

    private void testInvalidScaleModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp?min_worker=10&max_worker=1");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(httpStatus, HttpResponseStatus.BAD_REQUEST);
            assertEquals(resp.getMessage(), "max_worker cannot be less than min_worker.");
        }
    }

    private void testScaleModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), "Model or workflow not found: fake");
        }
    }

    private void testUnregisterModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/fake");
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), "Model or workflow not found: fake");
        }
    }

    private void testServiceUnavailable() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        reset();
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=0&max_worker=0");
        channel.writeAndFlush(req);
        latch.await();

        reset();
        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp_2");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        channel.writeAndFlush(req).sync();
        latch.await();
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.SERVICE_UNAVAILABLE.code());
            assertEquals(resp.getMessage(), "All model workers has been shutdown: mlp_2");
        }
    }

    private boolean checkWorkflowRegistered(Channel channel, String workflowName)
            throws InterruptedException {
        for (int i = 0; i < 5; ++i) {
            String token = "";
            while (token != null) {
                reset();
                HttpRequest req =
                        new DefaultFullHttpRequest(
                                HttpVersion.HTTP_1_1,
                                HttpMethod.GET,
                                "/workflows?limit=1&next_page_token=" + token);
                channel.writeAndFlush(req);
                latch.await();

                ListWorkflowsResponse resp =
                        JsonUtils.GSON.fromJson(result, ListWorkflowsResponse.class);
                for (WorkflowItem item : resp.getWorkflows()) {
                    if (workflowName.equals(item.getWorkflowName())) {
                        return true;
                    }
                }
                token = resp.getNextPageToken();
            }
            Thread.sleep(100);
        }
        return false;
    }

    private Channel connect(Connector.ConnectorType type) {
        Logger logger = LoggerFactory.getLogger(ModelServerTest.class);

        final Connector connector = configManager.getConnector(type);
        try {
            Bootstrap b = new Bootstrap();
            final SslContext sslCtx =
                    SslContextBuilder.forClient()
                            .trustManager(InsecureTrustManagerFactory.INSTANCE)
                            .build();
            EventLoopGroup elg = Connector.newEventLoopGroup(1);
            eventLoopGroups.add(elg);
            b.group(elg)
                    .channel(connector.getClientChannel())
                    .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, 10000)
                    .handler(
                            new ChannelInitializer<Channel>() {

                                /** {@inheritDoc} */
                                @Override
                                public void initChannel(Channel ch) {
                                    ChannelPipeline p = ch.pipeline();
                                    if (connector.isSsl()) {
                                        p.addLast(sslCtx.newHandler(ch.alloc()));
                                    }
                                    p.addLast(new ReadTimeoutHandler(30));
                                    p.addLast(new HttpClientCodec());
                                    p.addLast(new HttpContentDecompressor());
                                    p.addLast(new ChunkedWriteHandler());
                                    p.addLast(new HttpObjectAggregator(6553600));
                                    p.addLast(new TestHandler());
                                }
                            });

            return b.connect(connector.getSocketAddress()).sync().channel();
        } catch (Throwable t) {
            logger.warn("Connect error.", t);
        }
        throw new AssertionError("Failed connect to model server.");
    }

    private void reset() {
        result = null;
        httpStatus = null;
        headers = null;
        latch = new CountDownLatch(1);
    }

    @ChannelHandler.Sharable
    private class TestHandler extends SimpleChannelInboundHandler<FullHttpResponse> {

        /** {@inheritDoc} */
        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            latch.countDown();
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            latch.countDown();
        }
    }
}
