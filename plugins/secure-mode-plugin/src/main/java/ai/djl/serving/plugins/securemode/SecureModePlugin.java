package ai.djl.serving.plugins.securemode;

import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.wlm.util.EventManager;

import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class SecureModePlugin implements RequestHandler<Void> {

    private static final Logger LOGGER = LoggerFactory.getLogger(SecureModePlugin.class);

    public SecureModePlugin() {
        LOGGER.info("Register SecureModePlugin");
        LOGGER.info("Adding SecureModeModelServerListener");
        EventManager.getInstance().addListener(new SecureModeModelServerListener());
    }

    /** {@inheritDoc} */
    @Override
    public boolean acceptInboundMessage(Object msg) {
        return false;
    }

    /** {@inheritDoc} */
    @Override
    public Void handleRequest(
            ChannelHandlerContext ctx,
            FullHttpRequest req,
            QueryStringDecoder decoder,
            String[] segments) {
        return null;
    }
}
