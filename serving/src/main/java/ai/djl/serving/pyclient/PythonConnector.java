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

    private static final int MAX_BUFFER_SIZE = 6553500;
    private boolean uds;
    private String host;
    private int port;
    private String socketPath;

    /**
     * Constructs a {@code PythonConnector}.
     *
     * @param uds isuds
     * @param host host name for tcp connection
     * @param port port num for tcp connection
     * @param socketPath socket path for uds
     */
    public PythonConnector(boolean uds, String host, int port, String socketPath) {
        this.uds = uds;
        this.host = host;
        this.port = port;
        this.socketPath = socketPath;
    }

    /**
     * Returns whether connection can be uds or not.
     *
     * @return is uds
     */
    public boolean isUds() {
        return uds;
    }

    /**
     * Returns the host name.
     *
     * @return host
     */
    public String getHost() {
        return host;
    }

    /**
     * Returns the port number.
     *
     * @return port
     */
    public int getPort() {
        return port;
    }

    /**
     * Returns the socket path.
     *
     * @return socket path
     */
    public String getSocketPath() {
        return socketPath;
    }

    /**
     * Connects to the python server.
     *
     * @return netty client.
     */
    public Channel connect() {
        EventLoopGroup group = Connector.newEventLoopGroup(1);

        try {
            Bootstrap clientBootstrap = new Bootstrap();
            clientBootstrap.group(group);
            Class<? extends Channel> channelClass = Connector.getClientChannel(uds);
            clientBootstrap.channel(channelClass);
            clientBootstrap.remoteAddress(getSocketAddress());

            clientBootstrap.handler(
                    new ChannelInitializer<Channel>() {

                        @Override
                        protected void initChannel(Channel ch) {
                            ch.pipeline().addLast("decoder", new ResponseDecoder(MAX_BUFFER_SIZE));
                            ch.pipeline().addLast("handler", new RequestHandler());
                        }
                    });

            ChannelFuture future = clientBootstrap.connect().sync();
            return future.awaitUninterruptibly().channel();
        } catch (InterruptedException e) {
            logger.error("Exception occurred while creating netty client", e);
        }

        return null;
    }

    private SocketAddress getSocketAddress() {
        return uds
                ? new DomainSocketAddress(this.socketPath)
                : InetSocketAddress.createUnresolved(this.host, this.port);
    }
}
