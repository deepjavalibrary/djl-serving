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
package ai.djl.serving.http;

import io.netty.bootstrap.ServerBootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFutureListener;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.ServerChannel;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.epoll.EpollServerDomainSocketChannel;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.kqueue.KQueueServerDomainSocketChannel;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.http.DefaultFullHttpResponse;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpServerCodec;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.util.CharsetUtil;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** A small HTTP server on a Unix domain socket. It needs Epoll or KQueue. */
final class UnixSocketTestServer implements AutoCloseable {

    private final EventLoopGroup group;
    private final Channel channel;
    private final Path dir;
    private final String path;
    private final List<String> requests = new CopyOnWriteArrayList<>();
    private final AtomicInteger connections = new AtomicInteger();

    private UnixSocketTestServer(Responder responder) throws IOException, InterruptedException {
        dir = Files.createTempDirectory("djlem");
        path = dir.resolve("worker.metrics").toString();
        Class<? extends ServerChannel> channelClass;
        if (Epoll.isAvailable()) {
            group = new EpollEventLoopGroup(1);
            channelClass = EpollServerDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            group = new KQueueEventLoopGroup(1);
            channelClass = KQueueServerDomainSocketChannel.class;
        } else {
            throw new IllegalStateException("Unix domain sockets require Epoll or KQueue");
        }
        ServerBootstrap bootstrap =
                new ServerBootstrap()
                        .group(group)
                        .channel(channelClass)
                        .childHandler(
                                new ChannelInitializer<Channel>() {
                                    @Override
                                    protected void initChannel(Channel ch) {
                                        connections.incrementAndGet();
                                        ch.pipeline()
                                                .addLast(new HttpServerCodec())
                                                .addLast(new HttpObjectAggregator(1 << 20))
                                                .addLast(new Handler(responder));
                                    }
                                });
        channel = bootstrap.bind(new DomainSocketAddress(path)).sync().channel();
    }

    static boolean isSupported() {
        return Epoll.isAvailable() || KQueue.isAvailable();
    }

    static UnixSocketTestServer start(Responder responder)
            throws IOException, InterruptedException {
        return new UnixSocketTestServer(responder);
    }

    /** Answers {@code 200} with a Prometheus text body. */
    static Responder ok(String body) {
        return (ctx, request) -> reply(ctx, HttpResponseStatus.OK, body);
    }

    /** Answers with the given status and body. */
    static Responder status(HttpResponseStatus status, String body) {
        return (ctx, request) -> reply(ctx, status, body);
    }

    /** Answers {@code 200} after a delay. */
    static Responder delayed(long millis, String body) {
        return (ctx, request) ->
                ctx.executor()
                        .schedule(
                                () -> reply(ctx, HttpResponseStatus.OK, body),
                                millis,
                                TimeUnit.MILLISECONDS);
    }

    /** Writes the text as it is, without HTTP framing, and closes the connection. */
    static Responder raw(String text) {
        return (ctx, request) -> {
            ctx.pipeline().remove(HttpServerCodec.class);
            ctx.writeAndFlush(Unpooled.copiedBuffer(text, CharsetUtil.UTF_8))
                    .addListener(ChannelFutureListener.CLOSE);
        };
    }

    /** Reads the request and never answers. */
    static Responder silent() {
        return (ctx, request) -> {};
    }

    /** Closes the connection without answering. */
    static Responder closing() {
        return (ctx, request) -> ctx.close();
    }

    /** Returns the path of the socket. */
    String path() {
        return path;
    }

    /** Returns one line per request: method, target, Host and Connection headers. */
    List<String> requests() {
        return new ArrayList<>(requests);
    }

    /** Returns the number of accepted connections. */
    int connections() {
        return connections.get();
    }

    /** {@inheritDoc} */
    @Override
    public void close() throws IOException {
        channel.close().syncUninterruptibly();
        group.shutdownGracefully(0, 1, TimeUnit.SECONDS).syncUninterruptibly();
        Files.deleteIfExists(Path.of(path));
        Files.deleteIfExists(dir);
    }

    private static void reply(ChannelHandlerContext ctx, HttpResponseStatus status, String body) {
        byte[] bytes = body.getBytes(CharsetUtil.UTF_8);
        FullHttpResponse response =
                new DefaultFullHttpResponse(
                        HttpVersion.HTTP_1_1, status, Unpooled.wrappedBuffer(bytes));
        response.headers()
                .set(HttpHeaderNames.CONTENT_TYPE, "text/plain; version=0.0.4; charset=utf-8")
                .set(HttpHeaderNames.CONTENT_LENGTH, bytes.length);
        ctx.writeAndFlush(response).addListener(ChannelFutureListener.CLOSE);
    }

    /** Produces the answer to one request. */
    interface Responder {

        void respond(ChannelHandlerContext ctx, FullHttpRequest request);
    }

    private final class Handler extends SimpleChannelInboundHandler<FullHttpRequest> {

        private final Responder responder;

        Handler(Responder responder) {
            this.responder = responder;
        }

        /** {@inheritDoc} */
        @Override
        protected void channelRead0(ChannelHandlerContext ctx, FullHttpRequest request) {
            requests.add(
                    request.method()
                            + " "
                            + request.uri()
                            + " | "
                            + request.headers().get(HttpHeaderNames.HOST)
                            + " | "
                            + request.headers().get(HttpHeaderNames.CONNECTION));
            responder.respond(ctx, request);
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            ctx.close();
        }
    }
}
