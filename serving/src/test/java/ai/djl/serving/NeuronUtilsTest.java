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
package ai.djl.serving;

import ai.djl.serving.util.NeuronUtils;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.Proxy;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLStreamHandler;
import java.net.URLStreamHandlerFactory;
import java.nio.charset.StandardCharsets;
import org.testng.Assert;
import org.testng.annotations.Test;

public class NeuronUtilsTest {

    int mockMode;

    @Test
    public void testNeuronUtils() {
        MockURLStreamHandlerFactory factory = new MockURLStreamHandlerFactory();

        try {
            URL.setURLStreamHandlerFactory(factory);
            factory.setMock(true);

            mockMode = 0; // inf1.2xlarge
            boolean hasNeuron = NeuronUtils.hasNeuron();
            Assert.assertTrue(hasNeuron);

            mockMode = 1; // inf1.6xlarge
            hasNeuron = NeuronUtils.hasNeuron();
            Assert.assertTrue(hasNeuron);

            mockMode = 2; // inf1.24.xlarge
            hasNeuron = NeuronUtils.hasNeuron();
            Assert.assertTrue(hasNeuron);

            mockMode = 3; // c5.xlarge
            hasNeuron = NeuronUtils.hasNeuron();
            Assert.assertFalse(hasNeuron);

            mockMode = 4; // non EC2
            hasNeuron = NeuronUtils.hasNeuron();
            Assert.assertFalse(hasNeuron);
        } finally {
            factory.setMock(false);
        }
    }

    final class MockURLStreamHandlerFactory implements URLStreamHandlerFactory {

        private boolean mock;

        public void setMock(boolean mock) {
            this.mock = mock;
        }

        /** {@inheritDoc} */
        @Override
        public URLStreamHandler createURLStreamHandler(String protocol) {
            if (!mock) {
                return null;
            }
            return new URLStreamHandler() {

                /** {@inheritDoc} */
                @Override
                protected URLConnection openConnection(URL u, Proxy proxy) {
                    return openConnection(u);
                }

                /** {@inheritDoc} */
                @Override
                protected URLConnection openConnection(URL u) {
                    return new MockHttpURLConnection(u);
                }
            };
        }

        final class MockHttpURLConnection extends HttpURLConnection {

            public MockHttpURLConnection(URL u) {
                super(u);
            }

            /** {@inheritDoc} */
            @Override
            public void disconnect() {}

            /** {@inheritDoc} */
            @Override
            public boolean usingProxy() {
                return true;
            }

            /** {@inheritDoc} */
            @Override
            public void connect() {}

            /** {@inheritDoc} */
            @Override
            public InputStream getInputStream() throws IOException {
                switch (mockMode) {
                    case 0:
                        // EC2 inf1.2xlarge
                        return new ByteArrayInputStream(
                                "inf1.2xlarge".getBytes(StandardCharsets.UTF_8));
                    case 1:
                        // EC2 inf1.6xlarge
                        return new ByteArrayInputStream(
                                "inf1.6xlarge".getBytes(StandardCharsets.UTF_8));
                    case 2:
                        // EC2 inf1.24xlarge
                        return new ByteArrayInputStream(
                                "inf1.24xlarge".getBytes(StandardCharsets.UTF_8));
                    case 3:
                        // EC2 c5.xlarge
                        return new ByteArrayInputStream(
                                "c5.xlarge".getBytes(StandardCharsets.UTF_8));
                    default:
                        // non-AWS
                        throw new IOException("Timeout");
                }
            }

            /** {@inheritDoc} */
            @Override
            public int getResponseCode() {
                return HttpURLConnection.HTTP_OK;
            }
        }
    }
}
