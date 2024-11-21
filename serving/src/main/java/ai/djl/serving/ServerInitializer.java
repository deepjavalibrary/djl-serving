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

import ai.djl.serving.http.AdapterManagementRequestHandler;
import ai.djl.serving.http.ClusterRequestHandler;
import ai.djl.serving.http.ConfigurableHttpRequestHandler;
import ai.djl.serving.http.InferenceRequestHandler;
import ai.djl.serving.http.InvalidRequestHandler;
import ai.djl.serving.http.ManagementRequestHandler;
import ai.djl.serving.plugins.FolderScanPluginManager;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;

import io.netty.channel.Channel;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.ssl.SslContext;

/**
 * A special {@link io.netty.channel.ChannelInboundHandler} which offers an easy way to initialize a
 * {@link io.netty.channel.Channel} once it was registered to its {@link
 * io.netty.channel.EventLoop}.
 */
public class ServerInitializer extends ChannelInitializer<Channel> {

    private Connector.ConnectorType connectorType;
    private SslContext sslCtx;
    private FolderScanPluginManager pluginManager;
    private final ClusterRequestHandler clusterRequestHandler;

    /**
     * Creates a new {@code HttpRequestHandler} instance.
     *
     * @param sslCtx null if SSL is not enabled
     * @param connectorType type of {@link Connector}
     * @param pluginManager a pluginManager instance.
     */
    public ServerInitializer(
            SslContext sslCtx,
            Connector.ConnectorType connectorType,
            FolderScanPluginManager pluginManager) {
        this.sslCtx = sslCtx;
        this.connectorType = connectorType;
        this.pluginManager = pluginManager;
        this.clusterRequestHandler = ClusterRequestHandler.getInstance();
    }

    /** {@inheritDoc} */
    @Override
    public void initChannel(Channel ch) {
        ChannelPipeline pipeline = ch.pipeline();
        int maxRequestSize = ConfigManager.getInstance().getMaxRequestSize();
        if (sslCtx != null) {
            pipeline.addLast("ssl", sslCtx.newHandler(ch.alloc()));
        }
        pipeline.addLast("http", new HttpServerCodec());
        pipeline.addLast("aggregator", new HttpObjectAggregator(maxRequestSize, true));
        switch (connectorType) {
            case MANAGEMENT:
                pipeline.addLast(new ConfigurableHttpRequestHandler(pluginManager));
                pipeline.addLast("management", new ManagementRequestHandler());
                pipeline.addLast("management-adapter", new AdapterManagementRequestHandler());
                break;
            case INFERENCE:
                pipeline.addLast("inference", new InferenceRequestHandler());
                break;
            case CLUSTER:
                pipeline.addLast("cluster", clusterRequestHandler);
                break;
            case BOTH:
            default:
                pipeline.addLast(new ConfigurableHttpRequestHandler(pluginManager));
                pipeline.addLast("inference", new InferenceRequestHandler());
                pipeline.addLast("management", new ManagementRequestHandler());
                pipeline.addLast("management-adapter", new AdapterManagementRequestHandler());
                break;
        }
        pipeline.addLast("badRequest", new InvalidRequestHandler());
    }
}
