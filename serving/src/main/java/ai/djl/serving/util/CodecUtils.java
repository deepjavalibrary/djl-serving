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
import ai.djl.util.PairList;

import java.io.*;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;

public class CodecUtils {

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
            throw new IOException("Error while encoding the processing file and package", exception);
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
        int BUFFER_SIZE = 81920;

        ByteArrayInputStream in = new ByteArrayInputStream(byteArray);
        try (DataInputStream dis = new DataInputStream(in)) {
            String requestId = dis.readUTF();
            int code = dis.readInt();
            String message = dis.readUTF();
            Output output = new Output(code, message);
            output.setRequestId(requestId);

            int prop_size = dis.readInt();
            for (int i = 0; i < prop_size; i++) {
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
}
