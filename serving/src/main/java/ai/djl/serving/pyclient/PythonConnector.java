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
package ai.djl.serving.pyclient;

import ai.djl.serving.pyclient.protocol.ResponseDecoder;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.Connector;
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.unix.DomainSocketAddress;
import java.net.InetSocketAddress;
import java.net.SocketAddress;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** This class creates a netty client. */
public class PythonConnector {

    private static final Logger logger = LoggerFactory.getLogger(PythonConnector.class);
    private static final PythonConnector PYTHON_CONNECTOR = newInstance();
    private static final String DEFAULT_SOCKET_PATH = "/tmp/uds_sock";

    private static final int MAX_BUFFER_SIZE = 6553500;
    private Channel channel;

    /** Creates a netty client. */
    public PythonConnector() {
        EventLoopGroup group = Connector.newEventLoopGroup(1);

        try {
            Bootstrap clientBootstrap = new Bootstrap();
            // TODO: Support uds also
            clientBootstrap.group(group);
            Class<? extends Channel> channelClass = Connector.getClientChannel(isUDS());
            clientBootstrap.channel(channelClass);
            clientBootstrap.remoteAddress(getSocketAddress());

            clientBootstrap.handler(
                    new ChannelInitializer<Channel>() {
                        @Override
                        protected void initChannel(Channel ch) throws Exception {
                            ch.pipeline().addLast("decoder", new ResponseDecoder(MAX_BUFFER_SIZE));
                            ch.pipeline().addLast("handler", new RequestHandler());
                        }
                    });

            ChannelFuture future = clientBootstrap.connect().sync();
            this.channel = future.awaitUninterruptibly().channel();
            future.channel().closeFuture();
        } catch (Exception exception) {
            logger.error("Exception occurred while creating netty client", exception);
        }
    }

    /**
     * Getter for socket connector instance.
     *
     * @return socket connector instance
     */
    public static PythonConnector getInstance() {
        return PYTHON_CONNECTOR;
    }

    /**
     * Getter for netty client channel.
     *
     * @return channel for netty client.
     */
    public Channel getChannel() {
        return this.channel;
    }

    private static PythonConnector newInstance() {
        return new PythonConnector();
    }

    private boolean isUDS() {
        ConfigManager configManager = ConfigManager.getInstance();
        return configManager.useNativeIo();
    }

    private SocketAddress getSocketAddress() {
        return isUDS()
                ? new DomainSocketAddress(DEFAULT_SOCKET_PATH)
                : InetSocketAddress.createUnresolved("127.0.0.1", 9000);
    }
}
