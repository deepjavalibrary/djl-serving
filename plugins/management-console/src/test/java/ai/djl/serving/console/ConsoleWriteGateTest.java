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
package ai.djl.serving.console;

import ai.djl.serving.http.MethodNotAllowedException;

import io.netty.handler.codec.http.DefaultFullHttpRequest;
import io.netty.handler.codec.http.FullHttpRequest;
import io.netty.handler.codec.http.HttpMethod;
import io.netty.handler.codec.http.HttpVersion;
import io.netty.handler.codec.http.QueryStringDecoder;

import org.testng.Assert;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** Tests that state-changing console operations refuse unless explicitly enabled. */
public class ConsoleWriteGateTest {

    private static final String KEY = "DJL_ENABLE_CONSOLE_WRITE";

    @AfterMethod
    public void tearDown() {
        System.clearProperty(KEY);
    }

    @Test
    public void testWriteIsRefusedWhenUnset() {
        System.clearProperty(KEY);
        Assert.assertFalse(ConsoleRequestHandler.isConsoleWriteEnabled());
        Assert.assertThrows(
                MethodNotAllowedException.class, ConsoleRequestHandler::requiresWriteEnabled);
    }

    @DataProvider(name = "refusedValues")
    public Object[][] refusedValues() {
        return new Object[][] {{""}, {"false"}, {"0"}, {"no"}, {"yes"}, {"1"}, {"on"}};
    }

    @Test(dataProvider = "refusedValues")
    public void testWriteIsRefusedForAnythingButTrue(String value) {
        // Boolean.parseBoolean, so only "true" enables it. Spelled out because the Secure Mode
        // control deliberately accepts pydantic's wider set and these must not be confused.
        System.setProperty(KEY, value);
        Assert.assertFalse(ConsoleRequestHandler.isConsoleWriteEnabled(), value);
        Assert.assertThrows(
                MethodNotAllowedException.class, ConsoleRequestHandler::requiresWriteEnabled);
    }

    @DataProvider(name = "acceptedValues")
    public Object[][] acceptedValues() {
        return new Object[][] {{"true"}, {"TRUE"}, {"True"}};
    }

    @Test(dataProvider = "acceptedValues")
    public void testWriteIsPermittedWhenEnabled(String value) {
        System.setProperty(KEY, value);
        Assert.assertTrue(ConsoleRequestHandler.isConsoleWriteEnabled(), value);
        ConsoleRequestHandler.requiresWriteEnabled();
    }

    @DataProvider(name = "writeRoutes")
    public Object[][] writeRoutes() {
        return new Object[][] {
            {HttpMethod.POST, "/console/api/upload"},
            {HttpMethod.GET, "/console/api/restart"},
            {HttpMethod.POST, "/console/api/config"},
            {HttpMethod.POST, "/console/api/dependency"},
            {HttpMethod.DELETE, "/console/api/dependency/some.jar"},
        };
    }

    /**
     * Dispatches each write route to prove the gate is reached, not merely that it works.
     *
     * <p>The context is null: every one of these must refuse before it touches the channel, so a
     * route that reached its handler would fail here rather than quietly pass.
     *
     * @param method the request method
     * @param uri the request uri
     */
    @Test(dataProvider = "writeRoutes")
    public void testWriteRoutesRefuseThroughDispatch(HttpMethod method, String uri) {
        System.clearProperty(KEY);
        Assert.assertThrows(MethodNotAllowedException.class, () -> dispatch(method, uri));
    }

    @Test
    public void testReadRouteIsNotGated() {
        // The should-pass cell for the wiring: a read route must not be refused by the write
        // gate. version only writes a response, so a null context is what it fails on instead.
        System.clearProperty(KEY);
        Assert.assertThrows(
                NullPointerException.class, () -> dispatch(HttpMethod.GET, "/console/api/version"));
    }

    private static void dispatch(HttpMethod method, String uri) {
        FullHttpRequest req = new DefaultFullHttpRequest(HttpVersion.HTTP_1_1, method, uri);
        try {
            new ConsoleRequestHandler()
                    .handleRequest(null, req, new QueryStringDecoder(uri), uri.split("/", -1));
        } finally {
            req.release();
        }
    }
}
