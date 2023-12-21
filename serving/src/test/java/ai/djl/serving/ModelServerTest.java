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

import ai.djl.engine.Engine;
import ai.djl.modality.Classifications.Classification;
import ai.djl.repository.MRL;
import ai.djl.repository.Repository;
import ai.djl.serving.http.DescribeAdapterResponse;
import ai.djl.serving.http.DescribeWorkflowResponse;
import ai.djl.serving.http.DescribeWorkflowResponse.Model;
import ai.djl.serving.http.ErrorResponse;
import ai.djl.serving.http.ServerStartupException;
import ai.djl.serving.http.StatusResponse;
import ai.djl.serving.http.list.ListAdaptersResponse;
import ai.djl.serving.http.list.ListModelsResponse;
import ai.djl.serving.http.list.ListWorkflowsResponse;
import ai.djl.serving.http.list.ListWorkflowsResponse.WorkflowItem;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import ai.djl.util.JsonUtils;
import ai.djl.util.Utils;
import ai.djl.util.ZipUtils;
import ai.djl.util.cuda.CudaUtils;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.reflect.TypeToken;

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.ByteBuf;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
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

import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.AfterSuite;
import org.testng.annotations.BeforeSuite;
import org.testng.annotations.Test;

import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Type;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import javax.net.ssl.HttpsURLConnection;
import javax.net.ssl.SSLContext;

public class ModelServerTest {

    private static final String ERROR_NOT_FOUND =
            "Requested resource is not found, please refer to API document.";
    private static final String ERROR_METHOD_NOT_ALLOWED =
            "Requested method is not allowed, please refer to API document.";

    private ConfigManager configManager;
    private byte[] testImage;
    private List<EventLoopGroup> eventLoopGroups;
    volatile CountDownLatch latch;
    volatile CountDownLatch latch2;
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
        String engineCacheDir = Utils.getEngineCacheDir().toString();
        System.setProperty("DJL_CACHE_DIR", "build/cache");
        System.setProperty("ENGINE_CACHE_DIR", engineCacheDir);

        // TODO Remove when removing the temporary feature flag
        System.setProperty("ENABLE_ADAPTERS_PREVIEW", "true");
    }

    @AfterSuite
    public void afterSuite() {
        System.clearProperty("DJL_CACHE_DIR");
        System.clearProperty("ENGINE_CACHE_DIR");
    }

    @AfterMethod
    public void afterMethod() {
        for (EventLoopGroup elg : eventLoopGroups) {
            elg.shutdownGracefully(0, 0, TimeUnit.SECONDS);
        }
        eventLoopGroups = new ArrayList<>();

        ModelManager.getInstance().clear();
    }

    public static void main(String[] args)
            throws ReflectiveOperationException, ServerStartupException, GeneralSecurityException,
                    ErrorDataEncoderException, IOException, ParseException, InterruptedException {
        ModelServerTest t = new ModelServerTest();
        t.beforeSuite();
        t.test();
        t.afterMethod();
        t.afterSuite();
    }

    @Test
    public void test()
            throws InterruptedException, HttpPostRequestEncoder.ErrorDataEncoderException,
                    IOException, ParseException, GeneralSecurityException,
                    ReflectiveOperationException, ServerStartupException {
        ModelServer server = initTestServer("src/test/resources/config.properties");
        try {
            Path notModel = Paths.get("build/non-model");
            String url = server.mapModelUrl(notModel); // not a model dir
            assertNull(url);

            assertTrue(server.isRunning());
            Channel channel = initTestChannel();

            assertNotNull(channel, "Failed to connect to inference port.");

            // KServe v2 tests
            testKServeDescribeModel(channel);
            testKServeV2HealthLive(channel);
            testKServeV2HealthReady(channel);
            testKServeV2ModelReady(channel);
            testKServeV2Infer(channel);

            // inference API
            testRegisterModelTranslator(channel);
            testPing(channel);
            testRoot(channel);
            testPredictionsModels(channel);
            testInvoke(channel);
            testInvocations(channel);
            testInvocationsMultipart(channel);
            testDescribeApi(channel);

            // management API
            testRegisterModel(channel);
            testRegisterModelUnencoded(channel);
            testPerModelWorkers(channel);
            testRegisterModelAsync(channel);
            testRegisterWorkflow(channel);
            testRegisterWorkflowAsync(channel);
            testScaleModel(channel);
            testListModels(channel);
            testListWorkflows(channel);
            testDescribeModel(channel);
            testUnregisterModel(channel);
            testAsyncInference(channel);
            testThrottle(channel);
            testDjlModelZoo(channel);

            testPredictionsInvalidRequestSize(channel);

            // adapter API
            testAdapterRegister(channel);
            testAdapterNoPredictRegister();
            testAdapterPredict(channel);
            testAdapterDirPredict(channel);
            testAdapterInvoke(channel);
            testAdapterList(channel);
            testAdapterDescribe(channel);
            testAdapterScale(channel);
            testAdapterUnregister(channel);

            // plugin tests
            testStaticHtmlRequest();

            channel.close().sync();

            // negative test case that channel will be closed by server
            testInvalidUri();
            testInvalidPredictionsUri();
            testInvalidPredictionsMethod();
            testPredictionsModelNotFound();
            testInvalidDescribeModel();
            testInvalidKServeDescribeModel();
            testDescribeModelNotFound();
            testInvalidManagementUri();
            testInvalidManagementMethod();
            testUnregisterModelNotFound();
            testScaleModelNotFound();
            testRegisterModelMissingUrl();
            testRegisterModelNotFound();
            testRegisterModelConflict();
            testServiceUnavailable();

            ConfigManagerTest.testSsl();
        } finally {
            server.stop();
        }
    }

    @Test
    public void testWorkflows()
            throws ServerStartupException, GeneralSecurityException, ParseException, IOException,
                    InterruptedException, ReflectiveOperationException {
        ModelServer server = initTestServer("src/test/resources/workflow.config.properties");
        try {
            assertTrue(server.isRunning());
            Channel channel = initTestChannel();

            testPredictions(channel, new String[] {"/predictions/m"});
            testPredictionsWorkflows(channel);

            channel.close().sync();

            ConfigManagerTest.testSsl();
        } finally {
            server.stop();
        }
    }

    @Test
    public void testAdapterWorkflows()
            throws ServerStartupException, GeneralSecurityException, ParseException, IOException,
                    InterruptedException, ReflectiveOperationException {
        ModelServer server =
                initTestServer("src/test/resources/adapterWorkflows/config.properties");
        try {
            assertTrue(server.isRunning());
            Channel channel = initTestChannel();

            testAdapterWorkflowPredict(channel, "adapter1", "a1");
            testAdapterWorkflowPredict(channel, "adapter2", "a2");

            channel.close().sync();

            ConfigManagerTest.testSsl();
        } finally {
            server.stop();
        }
    }

    @Test
    public void testInitEmptyModelStore()
            throws IOException, ServerStartupException, GeneralSecurityException, ParseException,
                    InterruptedException {
        Path modelStore = Paths.get("build/models");
        Utils.deleteQuietly(modelStore);
        Files.createDirectories(modelStore);
        ModelServer server = initTestServer("src/test/resources/emptyStore.config.properties");
        try {
            assertTrue(server.isRunning());
            Channel channel = initTestChannel();
            request(
                    channel,
                    new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models"));

            ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
            Assert.assertNull(resp.getNextPageToken());
            assertTrue(resp.getModels().isEmpty());
        } finally {
            server.stop();
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
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/"));

        assertEquals(result, "{}\n");
    }

    private void testPing(Channel channel) throws InterruptedException {
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/ping"));

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
            DefaultFullHttpRequest req =
                    new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, target);
            req.content().writeBytes(testImage);
            HttpUtil.setContentLength(req, req.content().readableBytes());
            req.headers()
                    .set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
            request(channel, req);

            Type type = new TypeToken<List<Classification>>() {}.getType();
            List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
            assertEquals(classifications.get(0).getClassName(), "0");
        }
    }

    private void testInvoke(Channel channel) throws InterruptedException {
        String url = "/models/mlp/invoke";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        request(channel, req);

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testInvocations(Channel channel) throws InterruptedException {
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/invocations");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        request(channel, req);

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

        Assert.assertTrue(latch.await(2, TimeUnit.MINUTES));

        Type type = new TypeToken<List<Classification>>() {}.getType();
        List<Classification> classifications = JsonUtils.GSON.fromJson(result, type);
        assertEquals(classifications.get(0).getClassName(), "0");
    }

    private void testRegisterModelAsync(Channel channel) throws InterruptedException {
        String url = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_1&synchronous=false&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse statusResp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(statusResp.getStatus(), "Model \"mlp_1\" registration scheduled.");

        assertTrue(checkWorkflowRegistered(channel, "mlp_1"));
    }

    private void testRegisterModel(Channel channel) throws InterruptedException {
        String url = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"mlp_2\" registered.");
    }

    private void testRegisterModelUnencoded(Channel channel) throws InterruptedException {
        String url = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=mlp_2_unencoded&url=" + url));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"mlp_2_unencoded\" registered.");
    }

    private void testPerModelWorkers(Channel channel) throws InterruptedException {
        String url = "file:src/test/resources/identity";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?url=" + URLEncoder.encode(url, StandardCharsets.UTF_8)));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/identity"));

        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        Type type = new TypeToken<DescribeWorkflowResponse[]>() {}.getType();
        DescribeWorkflowResponse[] resp = JsonUtils.GSON.fromJson(result, type);
        DescribeWorkflowResponse wf = resp[0];
        Model model = wf.getModels().get(0);
        assertEquals(model.getQueueSize(), 10);
        DescribeWorkflowResponse.Group group = model.getWorkGroups().get(0);

        assertEquals(group.getMinWorkers(), 2);
        assertEquals(group.getMaxWorkers(), CudaUtils.hasCuda() ? 3 : 4);

        // unregister identity model
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/identity"));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testRegisterModelTranslator(Channel channel)
            throws InterruptedException, IOException {
        String url =
                "https://resources.djl.ai/demo/pytorch/traced_resnet18.zip?"
                        + "translator=ai.djl.serving.translator.TestTranslator"
                        + "&topK=1";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=res18&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"res18\" registered.");

        // send request
        String imgUrl = "https://resources.djl.ai/images/kitten.jpg";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/res18");
        req.content().writeBytes(imgUrl.getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);

        assertEquals(result, "topK: 1, best: n02124075 Egyptian cat");

        Repository repo = Repository.newInstance("tmp", url);
        MRL mrl = repo.getResources().get(0);
        Path modelDir = repo.getResourceDirectory(mrl.getDefaultArtifact());
        Assert.assertTrue(Files.exists(modelDir));

        // Unregister model
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/res18"));

        resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model or workflow \"res18\" unregistered");

        // make sure cache directory is removed
        Assert.assertFalse(Files.exists(modelDir));
    }

    private void testRegisterWorkflow(Channel channel) throws InterruptedException {
        String url = "https://resources.djl.ai/test-models/basic-serving-workflow.json";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?url=" + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Workflow \"BasicWorkflow\" registered.");
    }

    private void testRegisterWorkflowAsync(Channel channel) throws InterruptedException {
        String url = "https://resources.djl.ai/test-models/basic-serving-workflow2.json";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/workflows?synchronous=false&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse statusResp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(statusResp.getStatus(), "Workflow \"BasicWorkflow2\" registration scheduled.");

        assertTrue(checkWorkflowRegistered(channel, "BasicWorkflow2"));
    }

    private void testScaleModel(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=2&max_worker=4"));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(
                resp.getStatus(),
                "Workflow \"mlp_2\" worker scaled. New Worker configuration min workers:2 max"
                        + " workers:4");
    }

    private void testListModels(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models"));

        ListModelsResponse resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
        Assert.assertNull(resp.getNextPageToken());
        for (String expectedModel : new String[] {"mlp", "mlp_1", "mlp_2"}) {
            assertTrue(
                    resp.getModels().stream()
                            .anyMatch(w -> expectedModel.equals(w.getModelName())));
        }

        // Test pure JSON works
        JsonObject rawResult = JsonUtils.GSON.fromJson(result, JsonObject.class);
        JsonArray rawModels = rawResult.get("models").getAsJsonArray();
        Set<String> modelProperties =
                new HashSet<>(Arrays.asList("modelName", "version", "modelUrl", "status"));
        for (JsonElement rrawModel : rawModels) {
            JsonObject rawModel = rrawModel.getAsJsonObject();
            assertTrue(modelProperties.containsAll(rawModel.keySet()));
        }

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models?limit=2"));
        resp = JsonUtils.GSON.fromJson(result, ListModelsResponse.class);
        Assert.assertEquals(resp.getNextPageToken(), "2");
    }

    private void testListWorkflows(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/workflows"));

        ListWorkflowsResponse resp = JsonUtils.GSON.fromJson(result, ListWorkflowsResponse.class);
        for (String expectedWorkflow : new String[] {"mlp", "mlp_1", "mlp_2", "BasicWorkflow"}) {
            assertTrue(
                    resp.getWorkflows().stream()
                            .anyMatch(w -> expectedWorkflow.equals(w.getWorkflowName())));
        }

        // Test pure JSON works
        JsonObject rawResult = JsonUtils.GSON.fromJson(result, JsonObject.class);
        assertTrue(rawResult.has("nextPageToken"));
        JsonArray rawWorkflows = rawResult.get("workflows").getAsJsonArray();
        Set<String> workflowProperties = new HashSet<>(Arrays.asList("workflowName", "version"));
        for (JsonElement rrawWorkflow : rawWorkflows) {
            JsonObject rawWorkflow = rrawWorkflow.getAsJsonObject();
            assertTrue(workflowProperties.containsAll(rawWorkflow.keySet()));
        }
    }

    private void testKServeDescribeModel(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/v2/models/mlp"));

        JsonElement json = JsonUtils.GSON.fromJson(result, JsonElement.class);
        JsonObject resp = json.getAsJsonObject();
        assertEquals(resp.get("name").getAsString(), "mlp");
        assertEquals(resp.get("platform").getAsString(), "pytorch_torchscript");
    }

    private void testDescribeModel(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/mlp_2"));

        Type type = new TypeToken<DescribeWorkflowResponse[]>() {}.getType();
        DescribeWorkflowResponse[] resp = JsonUtils.GSON.fromJson(result, type);
        DescribeWorkflowResponse wf = resp[0];
        assertEquals(wf.getWorkflowName(), "mlp_2");
        assertNull(wf.getVersion());

        List<Model> models = wf.getModels();
        Model model = models.get(0);
        assertEquals(model.getModelName(), "mlp_2");
        assertNotNull(model.getModelUrl());
        assertEquals(model.getBatchSize(), 1);
        assertEquals(model.getMaxBatchDelayMillis(), 100);
        assertEquals(model.getMaxIdleSeconds(), 60);
        assertEquals(model.getQueueSize(), 1000);
        assertTrue(model.getRequestInQueue() >= 0);
        assertEquals(model.getStatus(), "Healthy");
        assertFalse(model.isLoadedAtStartup());

        DescribeWorkflowResponse.Group group = model.getWorkGroups().get(0);
        boolean hasGpu = Engine.getEngine("PyTorch").getGpuCount() > 0;
        assertEquals(group.getDevice().isGpu(), hasGpu);
        assertEquals(group.getMinWorkers(), 2);
        assertEquals(group.getMaxWorkers(), 4);
        List<DescribeWorkflowResponse.Worker> workers = group.getWorkers();
        assertTrue(workers.size() > 1);

        DescribeWorkflowResponse.Worker worker = workers.get(0);
        assertTrue(worker.getId() > 0);
        assertNotNull(worker.getStartTime());
        assertNotNull(worker.getStatus());

        // Test pure JSON works
        JsonArray rawResult = JsonUtils.GSON.fromJson(result, JsonArray.class);
        JsonArray rawModels = rawResult.get(0).getAsJsonObject().get("models").getAsJsonArray();
        JsonObject rawModel = rawModels.get(0).getAsJsonObject();
        assertEquals(rawModel.get("modelName").getAsString(), "mlp_2");
    }

    private void testUnregisterModel(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/mlp_1"));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model or workflow \"mlp_1\" unregistered");
    }

    private void testAsyncInference(Channel channel) throws InterruptedException {
        String url = URLEncoder.encode("file:src/test/resources/echo", StandardCharsets.UTF_8);
        url = "/models?model_name=echo&job_queue_size=10&url=" + url;
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        // send request
        url = "/predictions/echo?stream=true&delay=1";
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.headers().add("x-synchronous", "false");
        request(channel, req);
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        String nextToken = headers.get("x-next-token");
        assertNotNull(nextToken);

        url = "/predictions/echo";
        String firstToken = "";
        int maxPoll = 10;
        while (headers.contains("x-next-token") && --maxPoll > 0) {
            nextToken = headers.get("x-next-token");
            req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
            req.headers().add("x-starting-token", nextToken);
            if (firstToken.isEmpty()) {
                req.headers().add("x-max-items", "1");
            }
            request(channel, req);
            assertTrue(httpStatus.code() < 300);
            if (firstToken.isEmpty()) {
                firstToken = result;
            }
            Thread.sleep(1000);
        }
        assertEquals(firstToken, "tok_0\n");

        // Unregister model
        url = "/models/echo";
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testDjlModelZoo(Channel channel) throws InterruptedException {
        String url = "src/test/resources/zoomodel";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=zoomodel&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(resp.getStatus(), "Model \"zoomodel\" registered.");
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/zoomodel"));
    }

    private void testThrottle(Channel channel) throws InterruptedException {
        String url = URLEncoder.encode("file:src/test/resources/echo", StandardCharsets.UTF_8);
        url = "/models?model_name=echo&url=" + url;
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        String url = "/predictions/echo?delay=1000";
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        reset();
        ChannelFuture f = channel.writeAndFlush(req);

        // send 2nd request use different connection
        latch2 = new CountDownLatch(1);
        Channel channel2 = connect(Connector.ConnectorType.MANAGEMENT, 1);
        url = "/predictions/echo";
        req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        channel2.writeAndFlush(req).sync();
        Assert.assertTrue(latch2.await(2, TimeUnit.MINUTES));
        if (CudaUtils.getGpuCount() <= 1) {
            // one request is not able to saturate workers in multi-GPU case
            assertEquals(httpStatus.code(), 503);
        }

        // wait for 1st response
        f.sync();
        Assert.assertTrue(latch.await(2, TimeUnit.MINUTES));

        // Unregister model
        url = "/models/echo";
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testAdapterRegister(Channel channel) throws InterruptedException {
        String url = URLEncoder.encode("file:src/test/resources/adaptecho", StandardCharsets.UTF_8);
        url = "/models?model_name=adaptecho&url=" + url;
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        // Test Missing Adapter before registering
        testAdapterMissing();

        url = "/models/adaptecho/adapters?name=" + "adaptable" + "&src=" + "src";
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testAdapterMissing() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        String url = "/predictions/adaptecho?adapter=adaptable";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes("testPredictAdapter".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            assertEquals(httpStatus.code(), 503);
        }
    }

    private void testAdapterNoPredictRegister() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        String url = "/predictions/adaptecho?adapter=adaptable";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.headers().set("handler", "register_adapter");
        req.headers().set("name", "malicious_name");
        req.headers().set("src", "malicious_url");
        req.content().writeBytes("tt".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            assertEquals(httpStatus.code(), 400);
        }
    }

    private void testAdapterPredict(Channel channel) throws InterruptedException {
        String url = "/predictions/adaptecho?adapter=adaptable";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes("testPredictAdapter".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertEquals(result, "adaptabletestPredictAdapter");
    }

    private void testAdapterDirPredict(Channel channel) throws InterruptedException {
        String url = "/predictions/adaptecho?adapter=myBuiltinAdapter";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes("testPredictBuiltinAdapter".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertEquals(result, "myBuiltinAdaptertestPredictBuiltinAdapter");
    }

    private void testAdapterWorkflowPredict(Channel channel, String workflow, String adapter)
            throws InterruptedException {
        String url = "/predictions/" + workflow;
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes("testAWP".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertEquals(result, adapter + "testAWP");
    }

    private void testAdapterInvoke(Channel channel) throws InterruptedException {
        String url = "/invocations?model_name=adaptecho&adapter=adaptable";
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, url);
        req.content().writeBytes("testInvokeAdapter".getBytes(StandardCharsets.UTF_8));
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.TEXT_PLAIN);
        request(channel, req);

        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertEquals(result, "adaptabletestInvokeAdapter");
    }

    private void testAdapterList(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/adaptecho/adapters"));

        ListAdaptersResponse resp = JsonUtils.GSON.fromJson(result, ListAdaptersResponse.class);
        assertTrue(resp.getAdapters().stream().anyMatch(a -> "adaptable".equals(a.getName())));
    }

    private void testAdapterDescribe(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.GET,
                        "/models/adaptecho/adapters/adaptable"));

        DescribeAdapterResponse resp =
                JsonUtils.GSON.fromJson(result, DescribeAdapterResponse.class);
        assertEquals(resp.getName(), "adaptable");
        assertEquals(resp.getSrc(), "src");
    }

    private void testAdapterScale(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/adaptecho?min_worker=4&max_worker=4"));

        StatusResponse resp = JsonUtils.GSON.fromJson(result, StatusResponse.class);
        assertEquals(
                resp.getStatus(),
                "Workflow \"adaptecho\" worker scaled. New Worker configuration min workers:4 max"
                        + " workers:4");

        // Runs prediction after scaling
        // Has 3/4 chance to hit a worker that was scaled
        testAdapterPredict(channel);
    }

    private void testAdapterUnregister(Channel channel) throws InterruptedException {
        String url = "/models/adaptecho/adapters/adaptable";
        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.DELETE, url));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testDescribeApi(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/mlp"));

        assertEquals(result, "{}\n");
    }

    private void testStaticHtmlRequest() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        request(channel, new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/"));

        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private void testPredictionsInvalidRequestSize(Channel channel) throws InterruptedException {
        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp");

        req.content().writeZero(11485760);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        request(channel, req);

        assertEquals(httpStatus, HttpResponseStatus.REQUEST_ENTITY_TOO_LARGE);
    }

    private void testInvalidUri() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl"));
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getMessage(), ERROR_NOT_FOUND);
        }
    }

    private void testInvalidKServeDescribeModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/v2/models/res"));
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            Type type = new TypeToken<HashMap<String, String>>() {}.getType();
            Map<String, String> map = JsonUtils.GSON.fromJson(result, type);
            assertEquals(httpStatus.code(), HttpResponseStatus.NOT_FOUND.code());
            String errorMsg = "Model not found: res";
            assertEquals(map.get("error"), errorMsg);
        }
    }

    private void testInvalidDescribeModel() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.INFERENCE);
        assertNotNull(channel);

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.OPTIONS, "/predictions/InvalidModel"));
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

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions"));
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

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/predictions/InvalidModel"));
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

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.GET, "/InvalidUrl"));
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

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models"));
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

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models/noop"));
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

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/models/InvalidModel"));
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

        HttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        request(channel, req);
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
        String uri = "/models?url=InvalidUrl";
        HttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, uri);
        request(channel, req);
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.NOT_FOUND.code());
            assertEquals(resp.getType(), "ModelNotFoundException");
        }
    }

    private void testRegisterModelConflict() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        String modelUrl = "djl://ai.djl.zoo/mlp/0.0.3/mlp";
        Map<String, String> map = Map.of("model_name", "mlp_2", "url", modelUrl);

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.POST, "/models");
        req.content().writeCharSequence(JsonUtils.GSON.toJson(map), StandardCharsets.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);

        request(channel, req);
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), 409);
            assertEquals(resp.getMessage(), "Workflow mlp_2 is already registered.");
        }
    }

    private void testScaleModelNotFound() throws InterruptedException {
        Channel channel = connect(Connector.ConnectorType.MANAGEMENT);
        assertNotNull(channel);

        request(
                channel,
                new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, HttpMethod.PUT, "/models/fake"));
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

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/fake"));
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

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.PUT,
                        "/models/mlp_2?min_worker=0&max_worker=0"));

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/predictions/mlp_2");
        req.content().writeBytes(testImage);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_OCTET_STREAM);
        request(channel, req);
        channel.closeFuture().sync();
        channel.close().sync();

        if (!System.getProperty("os.name").startsWith("Win")) {
            ErrorResponse resp = JsonUtils.GSON.fromJson(result, ErrorResponse.class);
            assertEquals(resp.getCode(), HttpResponseStatus.SERVICE_UNAVAILABLE.code());
            assertEquals(resp.getMessage(), "All model workers has been shutdown: mlp_2 (READY)");
        }
    }

    private void testKServeV2HealthReady(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/v2/health/ready"));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertTrue(headers.contains("x-request-id"));
    }

    private void testKServeV2HealthLive(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/v2/health/live"));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertTrue(headers.contains("x-request-id"));
    }

    private void testKServeV2ModelReady(Channel channel) throws InterruptedException {
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/v2/models/mlp/ready"));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
        assertTrue(headers.contains("x-request-id"));
    }

    private void testKServeV2Infer(Channel channel) throws InterruptedException {
        String url = "file:src/test/resources/identity";
        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1,
                        HttpMethod.POST,
                        "/models?model_name=identity&min_worker=1&url="
                                + URLEncoder.encode(url, StandardCharsets.UTF_8)));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        DefaultFullHttpRequest req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/v2/models/identity/infer");

        Map<String, String> output = new ConcurrentHashMap<>();
        output.put("name", "output0");
        Map<String, Object> input = new ConcurrentHashMap<>();
        input.put("name", "input0");
        input.put("datatype", "INT8");
        input.put("shape", new long[] {1, 10});
        input.put("data", new double[10]);
        Map<String, Object> data = new ConcurrentHashMap<>();
        data.put("id", "42");
        data.put("inputs", new Object[] {input});
        data.put("outputs", new Object[] {output});

        // trigger model metrics logging
        req.content().writeCharSequence(JsonUtils.GSON.toJson(data), StandardCharsets.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        request(channel, req);
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());

        req =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.POST, "/v2/models/identity/infer");
        req.content().writeCharSequence(JsonUtils.GSON.toJson(data), StandardCharsets.UTF_8);
        HttpUtil.setContentLength(req, req.content().readableBytes());
        req.headers().set(HttpHeaderNames.CONTENT_TYPE, HttpHeaderValues.APPLICATION_JSON);
        request(channel, req);

        request(
                channel,
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.DELETE, "/models/identity"));
        assertEquals(httpStatus.code(), HttpResponseStatus.OK.code());
    }

    private boolean checkWorkflowRegistered(Channel channel, String workflowName)
            throws InterruptedException {
        for (int i = 0; i < 5; ++i) {
            String token = "";
            while (token != null) {
                request(
                        channel,
                        new DefaultFullHttpRequest(
                                HttpVersion.HTTP_1_1,
                                HttpMethod.GET,
                                "/workflows?limit=1&next_page_token=" + token));

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

    private void request(Channel channel, HttpRequest req) throws InterruptedException {
        reset();
        channel.writeAndFlush(req).sync();
        Assert.assertTrue(latch.await(2, TimeUnit.MINUTES));
    }

    private Channel connect(Connector.ConnectorType type) {
        return connect(type, 0);
    }

    private Channel connect(Connector.ConnectorType type, int mode) {
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
                            new ChannelInitializer<>() {

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
                                    p.addLast(new TestHandler(mode));
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

        private int mode;

        TestHandler(int mode) {
            this.mode = mode;
        }

        /** {@inheritDoc} */
        @Override
        public void channelRead0(ChannelHandlerContext ctx, FullHttpResponse msg) {
            httpStatus = msg.status();
            result = msg.content().toString(StandardCharsets.UTF_8);
            headers = msg.headers();
            countDown();
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            Logger logger = LoggerFactory.getLogger(TestHandler.class);
            logger.error("Unknown exception", cause);
            ctx.close();
            countDown();
        }

        private void countDown() {
            if (mode == 0) {
                latch.countDown();
            } else {
                latch2.countDown();
            }
        }
    }
}
