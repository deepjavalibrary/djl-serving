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
package ai.djl.serving.http;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.mockito.MockitoAnnotations.openMocks;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.serving.Arguments;
import ai.djl.serving.util.ConfigManager;
import ai.djl.util.Utils;

import io.netty.channel.embedded.EmbeddedChannel;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpVersion;

import org.apache.commons.cli.CommandLine;
import org.mockito.ArgumentCaptor;
import org.mockito.Mock;
import org.mockito.invocation.InvocationOnMock;
import org.testng.annotations.AfterClass;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

public class ClusterRequestHandlerTest {
    private static final String TEST_DIRECTORY = "build/tmp/test-clusterrequesthandler";

    @Mock private ProcessBuilder mockProcessBuilder;
    @Mock private Process sshProcess;
    private String expectedKey;
    private String fileToDelete;
    private Random random = new Random();

    private ClusterRequestHandler handler = ClusterRequestHandler.getInstance();

    @BeforeClass
    public void setup() throws Exception {
        Path sshDir = Path.of(TEST_DIRECTORY).resolve(".ssh");
        Files.createDirectories(sshDir);
        handler.setSshGenDir(sshDir);

        ConfigManager.init(new Arguments(CommandLine.builder().build()));
    }

    @BeforeMethod
    public void setupTest() throws Exception {
        openMocks(this).close();
        handler.setProcessBuilderFunction(cmd -> mockProcessBuilder.command(cmd));
        setupMockProcessBuilder();
    }

    @AfterClass(alwaysRun = true)
    public void tearDownClass() {
        Utils.deleteQuietly(Path.of(TEST_DIRECTORY));
    }

    @AfterMethod(alwaysRun = true)
    public void tearDown() throws Exception {
        if (fileToDelete != null) {
            if (!new File(fileToDelete).delete()) {
                throw new IOException("Failed to delete file");
            }
        }
    }

    @Test
    public void testSshPublicKeyGeneration() {
        EmbeddedChannel channel = new EmbeddedChannel(handler);
        channel.writeInbound(
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/cluster/sshpublickey"));
        DefaultFullHttpResponse response = channel.readOutbound();
        assertEquals(response.status().code(), 200);
        byte[] content = new byte[response.content().readableBytes()];
        response.content().readBytes(content);
        channel.finish();

        assertEquals(new String(content, StandardCharsets.UTF_8), expectedKey);

        ArgumentCaptor<String[]> captor = ArgumentCaptor.forClass(String[].class);
        verify(mockProcessBuilder).command(captor.capture());
        assertEquals(captor.getValue()[0], "ssh-keygen");
    }

    @Test
    public void testSshPublicKeyGenerationRunOnlyOnce() throws Exception {
        EmbeddedChannel channel = new EmbeddedChannel(handler);
        channel.writeInbound(
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/cluster/sshpublickey"));
        DefaultFullHttpResponse response = channel.readOutbound();
        assertEquals(response.status().code(), 200);
        byte[] content = new byte[response.content().readableBytes()];
        response.content().readBytes(content);
        channel.finish();

        assertEquals(new String(content, StandardCharsets.UTF_8), expectedKey);

        channel = new EmbeddedChannel(handler);
        channel.writeInbound(
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/cluster/sshpublickey"));
        response = channel.readOutbound();
        assertEquals(response.status().code(), 200);
        content = new byte[response.content().readableBytes()];
        response.content().readBytes(content);
        channel.finish();

        assertEquals(new String(content, StandardCharsets.UTF_8), expectedKey);

        verify(mockProcessBuilder, times(1)).start();
    }

    @Test
    public void testSshKeyGenHandlesTimeout() throws Exception {
        doReturn(false).when(sshProcess).waitFor(anyLong(), any());
        EmbeddedChannel channel = new EmbeddedChannel(handler);
        channel.writeInbound(
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/cluster/sshpublickey"));
        DefaultFullHttpResponse response = channel.readOutbound();
        assertEquals(response.status().code(), 500);
        byte[] content = new byte[response.content().readableBytes()];
        response.content().readBytes(content);
        channel.finish();

        assertTrue(new String(content, StandardCharsets.UTF_8).contains("timeout"));
        verify(sshProcess).destroy();
    }

    @Test
    public void testSshKeyGenHandlesErrors() throws Exception {
        doReturn(1).when(sshProcess).exitValue();
        EmbeddedChannel channel = new EmbeddedChannel(handler);
        channel.writeInbound(
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, "/cluster/sshpublickey"));
        DefaultFullHttpResponse response = channel.readOutbound();
        assertEquals(response.status().code(), 500);
        byte[] content = new byte[response.content().readableBytes()];
        response.content().readBytes(content);
        channel.finish();

        assertTrue(new String(content, StandardCharsets.UTF_8).contains("failed"));
    }

    @Test
    public void testSshKeyGenThreadSafety() throws Exception {
        CountDownLatch runKeygen = new CountDownLatch(1);

        when(mockProcessBuilder.start())
                .thenAnswer(
                        i -> {
                            runKeygen.await();
                            return sshProcess;
                        });

        ExecutorService executor = Executors.newFixedThreadPool(10);
        List<Future<Void>> futures = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            futures.add(
                    executor.submit(
                            () -> {
                                EmbeddedChannel channel = new EmbeddedChannel(handler);
                                channel.writeInbound(
                                        new DefaultFullHttpRequest(
                                                HttpVersion.HTTP_1_1,
                                                HttpMethod.GET,
                                                "/cluster/sshpublickey"));
                                runKeygen.await();
                                Thread.sleep(1000);
                                DefaultFullHttpResponse response = channel.readOutbound();
                                byte[] content = new byte[response.content().readableBytes()];
                                response.content().readBytes(content);

                                assertEquals(
                                        response.status().code(),
                                        200,
                                        "Failed with "
                                                + new String(content, StandardCharsets.UTF_8));
                                channel.finish();
                                assertEquals(
                                        new String(content, StandardCharsets.UTF_8), expectedKey);
                                return null;
                            }));
        }

        runKeygen.countDown();
        executor.shutdown();
        for (Future<Void> future : futures) {
            try {
                future.get(3000, TimeUnit.MILLISECONDS);
            } catch (InterruptedException ex) {
                Thread.currentThread().interrupt();
            }
        }
        verify(mockProcessBuilder, times(1)).start();
    }

    @Test
    public void testClusterRequestHandlerIsSingleton() {
        ClusterRequestHandler requestHandler = ClusterRequestHandler.getInstance();
        ClusterRequestHandler requestHandler2 = ClusterRequestHandler.getInstance();
        assert requestHandler == requestHandler2;
    }

    private void setupMockProcessBuilder() throws Exception {
        byte[] sshKey = new byte[100];
        random.nextBytes(sshKey);
        expectedKey = "ssh-rsa " + new String(sshKey, StandardCharsets.UTF_8);
        when(mockProcessBuilder.start()).thenReturn(sshProcess);
        when(sshProcess.getInputStream())
                .thenReturn(
                        new ByteArrayInputStream("debugOutput".getBytes(StandardCharsets.UTF_8)));
        when(sshProcess.exitValue()).thenReturn(0);
        when(sshProcess.waitFor(anyLong(), any())).thenReturn(true);
        when(mockProcessBuilder.command(any(String[].class)))
                .thenAnswer(this::processCommandInvocation);
    }

    private ProcessBuilder processCommandInvocation(InvocationOnMock invocation)
            throws IOException {
        String publicFile = invocation.getArgument(invocation.getArguments().length - 1) + ".pub";
        try (BufferedWriter fileWriter = Files.newBufferedWriter(Path.of(publicFile))) {
            fileWriter.write(expectedKey);
        }
        fileToDelete = publicFile;
        return mockProcessBuilder;
    }
}
