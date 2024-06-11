package ai.djl.serving.plugins.securemode;

import ai.djl.serving.plugins.RequestHandler;
import ai.djl.serving.wlm.util.EventManager;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.QueryStringDecoder;

public class SecureModePlugin implements RequestHandler<Void> {
    
    public SecureModePlugin() {
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
