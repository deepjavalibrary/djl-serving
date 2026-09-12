/*
 * Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
import static org.testng.Assert.assertNull;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.bootstrap.ServerBootstrapConfig;
import io.netty.channel.ChannelOption;

import org.testng.annotations.Test;

/**
 * Regression coverage for {@link ModelServer#configureSocketOptions(ServerBootstrap, boolean)}.
 * Calls the real production method directly (rather than duplicating its option list in the test)
 * so a future edit to the socket options cannot silently drop {@code TCP_NODELAY} without failing
 * this test.
 */
public class ModelServerSocketOptionsTest {

    @Test
    public void testTcpNoDelayIsEnabledOnChildChannels() {
        ServerBootstrap b = new ServerBootstrap();
        ModelServer.configureSocketOptions(b, false);

        ServerBootstrapConfig config = b.config();
        assertEquals(
                config.childOptions().get(ChannelOption.TCP_NODELAY),
                Boolean.TRUE,
                "TCP_NODELAY must be enabled on accepted child channels: without it, Nagle's"
                        + " algorithm combined with client-side delayed ACK can stall each"
                        + " streamed HTTP chunk by tens of milliseconds, inflating inter-token"
                        + " latency and p99 for streaming/chat-completions responses");
    }

    @Test
    public void testTcpNoDelayIsNotSetOnUdsChannels() {
        ServerBootstrap b = new ServerBootstrap();
        ModelServer.configureSocketOptions(b, true);

        ServerBootstrapConfig config = b.config();
        assertNull(
                config.childOptions().get(ChannelOption.TCP_NODELAY),
                "TCP_NODELAY is a TCP-only option: domain-socket child channels don't support it,"
                        + " so setting it on a unix-socket bootstrap can break UDS"
                        + " initialization");
    }

    @Test
    public void testExistingSocketOptionsAreUnaffected() {
        ServerBootstrap b = new ServerBootstrap();
        ModelServer.configureSocketOptions(b, false);

        ServerBootstrapConfig config = b.config();
        assertEquals(config.options().get(ChannelOption.SO_BACKLOG), 1024);
        assertEquals(config.childOptions().get(ChannelOption.SO_LINGER), 0);
        assertEquals(config.childOptions().get(ChannelOption.SO_REUSEADDR), Boolean.TRUE);
        assertEquals(config.childOptions().get(ChannelOption.SO_KEEPALIVE), Boolean.TRUE);
    }

    @Test
    public void testConfigureSocketOptionsIsIdempotent() {
        ServerBootstrap b = new ServerBootstrap();
        ModelServer.configureSocketOptions(b, false);
        ModelServer.configureSocketOptions(b, false);

        assertEquals(
                b.config().childOptions().get(ChannelOption.TCP_NODELAY),
                Boolean.TRUE,
                "calling configureSocketOptions twice should not toggle the option off");
    }
}
