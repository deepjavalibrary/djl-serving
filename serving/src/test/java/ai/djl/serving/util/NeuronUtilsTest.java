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

import org.testng.Assert;
import org.testng.annotations.Test;

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

public class NeuronUtilsTest {

    @Test
    public void testNeuronUtils() {
        MockURLStreamHandlerFactory factory = new MockURLStreamHandlerFactory();

        try {
            URL.setURLStreamHandlerFactory(factory);

            NeuronUtils.setInstanceType(-1);
            factory.setMockMode(0); // inf1
            Assert.assertTrue(NeuronUtils.hasNeuron());
            Assert.assertTrue(NeuronUtils.isInf1());
            if (System.getProperty("os.name").startsWith("Linux")) {
                NeuronUtils.getNeuronCores();
            }

            NeuronUtils.setInstanceType(-1);
            factory.setMockMode(1); // inf2
            Assert.assertTrue(NeuronUtils.hasNeuron());
            Assert.assertTrue(NeuronUtils.isInf2());

            NeuronUtils.setInstanceType(-1);
            factory.setMockMode(2); // inf1
            Assert.assertTrue(NeuronUtils.hasNeuron());
            Assert.assertTrue(NeuronUtils.isInf2());

            NeuronUtils.setInstanceType(-1);
            factory.setMockMode(3); // inf1
            Assert.assertFalse(NeuronUtils.hasNeuron());

            NeuronUtils.setInstanceType(-1);
            factory.setMockMode(4); // inf1
            Assert.assertFalse(NeuronUtils.hasNeuron());
        } finally {
            factory.setMock(false);
        }
    }

    static final class MockURLStreamHandlerFactory implements URLStreamHandlerFactory {

        private boolean mock = true;
        private int mockMode;

        public void setMock(boolean mock) {
            this.mock = mock;
        }

        public void setMockMode(int mockMode) {
            this.mockMode = mockMode;
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
                        // EC2 inf2.24xlarge
                        return new ByteArrayInputStream(
                                "inf2.24xlarge".getBytes(StandardCharsets.UTF_8));
                    case 2:
                        // EC2 trn1.32xlarge
                        return new ByteArrayInputStream(
                                "trn1.32xlarge".getBytes(StandardCharsets.UTF_8));
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
