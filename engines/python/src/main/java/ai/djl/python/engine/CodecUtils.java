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
package ai.djl.python.engine;

import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.CorruptedFrameException;
import java.nio.charset.StandardCharsets;

/** This is a utility class for reading and writing to netty ByteBuf. */
public final class CodecUtils {

    public static final int MAX_BUFFER_SIZE = 20 * 1024 * 1024;

    private CodecUtils() {}

    /**
     * Reads the specified length of data.
     *
     * @param in byte buffer
     * @param maxLength length of the data to be read
     * @return read data
     */
    public static byte[] readBytes(ByteBuf in, int maxLength) {
        int len = in.readInt();
        if (len < 0) {
            return null;
        }
        if (len > maxLength) {
            throw new CorruptedFrameException("Message size exceed limit: " + len);
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }

    /**
     * Read a String from the {@code ByteBuf}.
     *
     * @param in the {@code ByteBuf}.
     * @return a string read from the buffer.
     */
    public static String readUtf8(ByteBuf in) {
        int len = in.readShort();
        if (len < 0) {
            return null;
        }
        byte[] buf = new byte[len];
        in.readBytes(buf);
        return new String(buf, StandardCharsets.UTF_8);
    }

    /**
     * Encode a String in UTF-8 and write it to the {@code ByteBuf}.
     *
     * @param buf the {@code ByteBuf}.
     * @param value the string to write into a buffer.
     */
    public static void writeUtf8(ByteBuf buf, String value) {
        if (value == null) {
            buf.writeShort(-1);
        } else {
            byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
            buf.writeShort(bytes.length);
            buf.writeBytes(bytes);
        }
    }
}
