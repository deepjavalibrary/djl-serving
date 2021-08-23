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
package ai.djl.serving.pyclient.protocol;

import ai.djl.serving.util.CodecUtils;
import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.ByteToMessageDecoder;
import java.util.List;

/** This class decodes the response from netty channel. */
public class ResponseDecoder extends ByteToMessageDecoder {

    private final int maxBufferSize;

    /**
     * Constructs a {@code ResponseDecoder} instance with the maximum buffer size.
     *
     * @param maxBufferSize limit of the buffer size that can be received.
     */
    public ResponseDecoder(int maxBufferSize) {
        this.maxBufferSize = maxBufferSize;
    }

    /** {@inheritDoc} */
    @Override
    protected void decode(ChannelHandlerContext ctx, ByteBuf in, List<Object> out) {
        int size = in.readableBytes();
        if (size < 4) {
            return;
        }

        // this index of the reader is marked,
        // so that future reads can be done from this index.
        in.markReaderIndex();
        boolean completed = false;

        try {
            int arrLen = CodecUtils.readLength(in, maxBufferSize);
            if (arrLen == CodecUtils.BUFFER_UNDER_RUN) {
                return;
            }

            byte[] arr = CodecUtils.read(in, arrLen);
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
