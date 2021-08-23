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
package ai.djl.serving.util;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.serving.pyclient.protocol.Request;
import ai.djl.util.PairList;
import io.netty.buffer.ByteBuf;
import io.netty.handler.codec.CorruptedFrameException;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;

/** This is a utility class for encoding and decoding of request and response with python server. */
public final class CodecUtils {

    public static final int BUFFER_SIZE = 81920;
    public static final int BUFFER_UNDER_RUN = -3;

    private CodecUtils() {}

    /**
     * Reads an integer value, which represents the length of the data.
     *
     * @param byteBuf byte buffer
     * @param maxLength maximum length of data that can be read.
     * @return length
     */
    public static int readLength(ByteBuf byteBuf, int maxLength) {
        int size = byteBuf.readableBytes();
        if (size < 4) {
            return BUFFER_UNDER_RUN;
        }

        int len = byteBuf.readInt();
        if (len > maxLength) {
            throw new CorruptedFrameException("Message size exceed limit: " + len);
        }
        if (len > byteBuf.readableBytes()) {
            return BUFFER_UNDER_RUN;
        }
        return len;
    }

    /**
     * Reads the specified length of data.
     *
     * @param in byte buffer
     * @param len length of the data to be read
     * @return read data
     */
    public static byte[] read(ByteBuf in, int len) {
        if (len < 0) {
            throw new CorruptedFrameException("Invalid message size: " + len);
        }

        byte[] buf = new byte[len];
        in.readBytes(buf);
        return buf;
    }

    /**
     * Converts the given Input to byte array.
     *
     * @param input generic input data for inference
     * @return byte array format of input
     * @throws IOException if error occurs while converting input to byte array
     */
    public static byte[] encodeInput(Input input) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream dos = new DataOutputStream(baos);
            dos.writeUTF(input.getRequestId());

            Map<String, String> properties = input.getProperties();
            dos.writeInt(properties.size());
            for (Map.Entry<String, String> entry : properties.entrySet()) {
                dos.writeUTF(entry.getKey());
                dos.writeUTF(entry.getValue());
            }

            PairList<String, byte[]> content = input.getContent();
            dos.writeInt(content.size());
            List<String> keys = content.keys();
            for (String key : keys) {
                if (key == null) {
                    dos.writeUTF("");
                } else {
                    dos.writeUTF(key);
                }
            }

            List<byte[]> values = content.values();
            for (byte[] value : values) {
                int length = value.length;
                dos.writeInt(length);
                dos.write(value);
            }
            dos.flush();
            return baos.toByteArray();
        } catch (IOException exception) {
            throw new IOException(
                    "Error while encoding the processing file and package", exception);
        }
    }

    /**
     * Decodes the bytearray into an Output.
     *
     * @param byteArray to be decoded to Output
     * @return output
     * @throws IOException when error occurs during decoding
     */
    public static Output decodeToOutput(byte[] byteArray) throws IOException {
        ByteArrayInputStream in = new ByteArrayInputStream(byteArray);
        try (DataInputStream dis = new DataInputStream(in)) {
            String requestId = dis.readUTF();
            int code = dis.readInt();
            String message = dis.readUTF();
            Output output = new Output(code, message);
            output.setRequestId(requestId);

            int propSize = dis.readInt();
            for (int i = 0; i < propSize; i++) {
                String key = dis.readUTF();
                String val = dis.readUTF();
                output.addProperty(key, val);
            }

            int length = dis.readInt();
            ByteBuffer data = ByteBuffer.allocate(length);

            if (length > 0) {
                byte[] buf = new byte[BUFFER_SIZE];
                while (length > BUFFER_SIZE) {
                    dis.readFully(buf);
                    data.put(buf);
                    length -= BUFFER_SIZE;
                }

                dis.readFully(buf, 0, length);
                data.put(buf, 0, length);
                data.rewind();
            }

            output.setContent(data.array());
            return output;
        } catch (IOException ioException) {
            throw new IOException("Exception while decoding output", ioException);
        }
    }

    /**
     * Encodes the request into byte array.
     *
     * @param request instance
     * @return encoded byte array
     * @throws IOException if error occurs while encoding
     */
    public static byte[] encodeRequest(Request request) throws IOException {
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream()) {
            DataOutputStream dos = new DataOutputStream(baos);
            dos.writeInt(request.getRequestType());
            dos.writeUTF(request.getPythonFile());
            dos.writeUTF(request.getFunctionName());
            dos.writeInt(request.getFunctionParam().length);
            dos.write(request.getFunctionParam());

            dos.flush();
            return baos.toByteArray();
        } catch (IOException ioException) {
            throw new IOException("Error while encoding Request", ioException);
        }
    }
}
