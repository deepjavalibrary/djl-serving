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
import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.SocketChannel;
import io.netty.channel.socket.nio.NioSocketChannel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class creates a netty client.
 */
public class SocketConnector {

    private static final Logger logger = LoggerFactory.getLogger(SocketConnector.class);
    private static final SocketConnector SOCKET_CONNECTOR = newInstance();

    private static final int MAX_BUFFER_SIZE = 6553500;
    private Channel channel;

    /**
     * Creates a netty client.
     */
    public SocketConnector() {

        EventLoopGroup group = new NioEventLoopGroup();

        try {
            Bootstrap clientBootstrap = new Bootstrap();
            //TODO: Support uds also
            clientBootstrap.group(group);
            clientBootstrap.channel(NioSocketChannel.class);
            clientBootstrap.remoteAddress("127.0.0.1", 9000);

            clientBootstrap.handler(new ChannelInitializer<SocketChannel>() {
                @Override
                protected void initChannel(SocketChannel socketChannel) throws Exception {
                    socketChannel.pipeline().addLast("decoder", new ResponseDecoder(MAX_BUFFER_SIZE));
                    socketChannel.pipeline().addLast("handler", new RequestHandler());
                }
            });

            ChannelFuture future = clientBootstrap.connect().sync();
            this.channel = future.awaitUninterruptibly().channel();
            future.channel().closeFuture();
        } catch (Exception exception) {
            logger.error("Exception occurred while creating netty client");
        }
    }

    /**
     * Getter for socket connector instance.
     *
     * @return socket connector instance
     */
    public static SocketConnector getInstance() {
        return SOCKET_CONNECTOR;
    }

    /**
     * Getter for netty client channel.
     *
     * @return channel for netty client.
     */
    public Channel getChannel() {
        return this.channel;
    }

    private static SocketConnector newInstance() {
        return new SocketConnector();
    }

}
