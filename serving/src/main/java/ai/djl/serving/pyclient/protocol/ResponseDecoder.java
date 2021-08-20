package ai.djl.serving.pyclient.protocol;

import ai.djl.serving.util.CodecUtils;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;

import java.util.List;

public class ResponseDecoder extends ByteToMessageDecoder {
    private final int maxBufferSize;

    public ResponseDecoder(int maxBufferSize) {
        this.maxBufferSize = maxBufferSize;
    }

    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) throws Exception {
        int size = in.readableBytes();
        if (size < 4) {
            return;
        }

        // this index of the reader is marked,
        // so that future reads can be done from this index.
        in.markReaderIndex();
        boolean completed = false;

        try {

            int arr_len = CodecUtils.readLength(in, maxBufferSize);
            if (arr_len == CodecUtils.BUFFER_UNDER_RUN) {
                return;
            }

            byte[] arr = CodecUtils.read(in, arr_len);
            Response response = new Response();
            response.setRawData(arr);
            completed = true;
            out.add(response);
        } finally {
            if (!completed) {
                // resetting the marked index. Index will be set to 0
                in.resetReaderIndex();
            }
        }
    }
}
