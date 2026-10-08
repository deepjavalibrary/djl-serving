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

import io.netty.bootstrap.Bootstrap;
import io.netty.buffer.Unpooled;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelOption;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.SimpleChannelInboundHandler;
import io.netty.channel.epoll.Epoll;
import io.netty.channel.epoll.EpollDomainSocketChannel;
import io.netty.channel.epoll.EpollEventLoopGroup;
import io.netty.channel.kqueue.KQueue;
import io.netty.channel.kqueue.KQueueDomainSocketChannel;
import io.netty.channel.kqueue.KQueueEventLoopGroup;
import io.netty.channel.unix.DomainSocketAddress;
import io.netty.handler.codec.DecoderException;
import io.netty.handler.codec.PrematureChannelClosureException;
import io.netty.handler.codec.TooLongFrameException;
import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.FullHttpResponse;
import io.netty.handler.codec.http.HttpClientCodec;
import io.netty.handler.codec.http.HttpHeaderNames;
import io.netty.handler.codec.http.HttpHeaderValues;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpObjectAggregator;
import io.netty.handler.codec.http.HttpResponseStatus;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.timeout.ReadTimeoutException;
import io.netty.handler.timeout.ReadTimeoutHandler;
import io.netty.util.CharsetUtil;
import io.netty.util.concurrent.DefaultThreadFactory;
import io.netty.util.concurrent.ScheduledFuture;

import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.ConnectException;
import java.nio.channels.ClosedChannelException;
import java.nio.file.NoSuchFileException;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * A minimal HTTP client that issues a single GET over a Unix domain socket.
 *
 * <p>All connections share one lazily created, single-thread, daemon event loop, so a slow or hung
 * peer never occupies an inference or management thread. The returned futures are completed on that
 * event loop. A request that cannot even be started fails on the calling thread.
 */
final class EngineMetricsClient {

    private static final int CONNECT_TIMEOUT_MILLIS = 1000;

    private static EventLoopGroup group;

    private EngineMetricsClient() {}

    /**
     * Returns {@code true} if this platform can connect to Unix domain sockets with netty.
     *
     * @return {@code true} if Epoll or KQueue is available
     */
    static boolean isSupported() {
        return Epoll.isAvailable() || KQueue.isAvailable();
    }

    /**
     * Sends {@code GET uri} to a Unix domain socket and returns the body of a {@code 200} answer.
     *
     * <p>The future fails with an {@link HttpStatusException} for any other status, a {@link
     * TimeoutException} when no complete answer arrived within {@code timeoutMs}, a {@link
     * ConnectException} when nothing accepts connections on the socket, a {@link
     * ClosedChannelException} or {@link PrematureChannelClosureException} when the peer closed
     * before the response was complete, a {@link TooLongFrameException} when the body exceeds
     * {@code maxBytes}, and an {@link IOException} when the response is malformed.
     *
     * @param socketPath the path of the Unix domain socket
     * @param uri the request target, for example {@code /metrics?worker=gpu0}
     * @param maxBytes the maximum size of the response body
     * @param timeoutMs the deadline for the whole exchange, including the connect
     * @return a future of the response body decoded as UTF-8
     */
    static CompletableFuture<String> get(
            String socketPath, String uri, int maxBytes, long timeoutMs) {
        CompletableFuture<String> result = new CompletableFuture<>();
        try {
            EventLoopGroup eventLoop = getGroup();
            Bootstrap bootstrap =
                    new Bootstrap()
                            .group(eventLoop)
                            .channel(getChannelClass())
                            .option(ChannelOption.CONNECT_TIMEOUT_MILLIS, CONNECT_TIMEOUT_MILLIS)
                            .handler(new Initializer(result, maxBytes, timeoutMs));
            ChannelFuture connected = bootstrap.connect(new DomainSocketAddress(socketPath));
            Channel channel = connected.channel();
            ScheduledFuture<?> deadline =
                    eventLoop.schedule(
                            () ->
                                    result.completeExceptionally(
                                            new TimeoutException(
                                                    "No answer within " + timeoutMs + " ms")),
                            timeoutMs,
                            TimeUnit.MILLISECONDS);
            result.whenComplete(
                    (body, error) -> {
                        deadline.cancel(false);
                        channel.close();
                    });
            connected.addListener(
                    f -> {
                        if (!f.isSuccess()) {
                            result.completeExceptionally(f.cause());
                            return;
                        }
                        FullHttpRequest request = newRequest(uri);
                        channel.writeAndFlush(request)
                                .addListener(
                                        w -> {
                                            if (!w.isSuccess()) {
                                                result.completeExceptionally(w.cause());
                                            }
                                        });
                    });
        } catch (RuntimeException e) {
            result.completeExceptionally(e);
        }
        return result;
    }

    /**
     * Returns a short human readable reason for a failed {@link #get} or for a failure of a source
     * checked before connecting.
     *
     * @param error the failure
     * @return the reason
     */
    static String describe(Throwable error) {
        Throwable cause = error;
        while ((cause instanceof CompletionException
                        || cause instanceof ExecutionException
                        || cause instanceof DecoderException)
                && cause.getCause() != null) {
            cause = cause.getCause();
        }
        if (cause instanceof TimeoutException || cause instanceof ReadTimeoutException) {
            return "timeout";
        } else if (cause instanceof HttpStatusException) {
            return "HTTP status " + ((HttpStatusException) cause).getStatus();
        } else if (cause instanceof NoSuchFileException || cause instanceof FileNotFoundException) {
            return "socket file missing";
        } else if (cause instanceof ConnectException) {
            return "cannot connect (" + cause.getMessage() + ")";
        } else if (cause instanceof ClosedChannelException
                || cause instanceof PrematureChannelClosureException) {
            return "closed before the response";
        } else if (cause instanceof TooLongFrameException) {
            return "response over the size limit";
        }
        String message = cause.getMessage();
        String name = cause.getClass().getSimpleName();
        return message == null ? name : name + ": " + message;
    }

    private static synchronized EventLoopGroup getGroup() {
        if (group == null) {
            ThreadFactory factory = new DefaultThreadFactory("djl-engine-metrics", true);
            if (Epoll.isAvailable()) {
                group = new EpollEventLoopGroup(1, factory);
            } else if (KQueue.isAvailable()) {
                group = new KQueueEventLoopGroup(1, factory);
            } else {
                throw new UnsupportedOperationException(
                        "Unix domain sockets require Epoll or KQueue");
            }
        }
        return group;
    }

    private static Class<? extends Channel> getChannelClass() {
        if (Epoll.isAvailable()) {
            return EpollDomainSocketChannel.class;
        } else if (KQueue.isAvailable()) {
            return KQueueDomainSocketChannel.class;
        }
        throw new UnsupportedOperationException("Unix domain sockets require Epoll or KQueue");
    }

    private static FullHttpRequest newRequest(String uri) {
        FullHttpRequest request =
                new DefaultFullHttpRequest(
                        HttpVersion.HTTP_1_1, HttpMethod.GET, uri, Unpooled.EMPTY_BUFFER);
        request.headers()
                .set(HttpHeaderNames.HOST, "localhost")
                .set(HttpHeaderNames.CONNECTION, HttpHeaderValues.CLOSE);
        return request;
    }

    /** Raised when the peer answers with a status other than {@code 200 OK}. */
    static final class HttpStatusException extends IOException {

        private static final long serialVersionUID = 1L;

        private final int status;

        HttpStatusException(HttpResponseStatus status) {
            super("HTTP " + status);
            this.status = status.code();
        }

        int getStatus() {
            return status;
        }
    }

    private static final class Initializer extends ChannelInitializer<Channel> {

        private final CompletableFuture<String> result;
        private final int maxBytes;
        private final long timeoutMs;

        Initializer(CompletableFuture<String> result, int maxBytes, long timeoutMs) {
            this.result = result;
            this.maxBytes = maxBytes;
            this.timeoutMs = timeoutMs;
        }

        /** {@inheritDoc} */
        @Override
        protected void initChannel(Channel ch) {
            ch.pipeline()
                    .addLast(new ReadTimeoutHandler(timeoutMs, TimeUnit.MILLISECONDS))
                    .addLast(new HttpClientCodec())
                    .addLast(new HttpObjectAggregator(maxBytes))
                    .addLast(new ResponseHandler(result));
        }
    }

    private static final class ResponseHandler
            extends SimpleChannelInboundHandler<FullHttpResponse> {

        private final CompletableFuture<String> result;

        ResponseHandler(CompletableFuture<String> result) {
            this.result = result;
        }

        /** {@inheritDoc} */
        @Override
        protected void channelRead0(ChannelHandlerContext ctx, FullHttpResponse response) {
            if (response.decoderResult().isFailure()) {
                result.completeExceptionally(
                        new IOException(
                                "Malformed HTTP response", response.decoderResult().cause()));
            } else if (HttpResponseStatus.OK.equals(response.status())) {
                result.complete(response.content().toString(CharsetUtil.UTF_8));
            } else {
                result.completeExceptionally(new HttpStatusException(response.status()));
            }
            ctx.close();
        }

        /** {@inheritDoc} */
        @Override
        public void channelInactive(ChannelHandlerContext ctx) {
            result.completeExceptionally(new ClosedChannelException());
            ctx.fireChannelInactive();
        }

        /** {@inheritDoc} */
        @Override
        public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
            result.completeExceptionally(cause);
            ctx.close();
        }
    }
}
