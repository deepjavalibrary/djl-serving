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
package ai.djl.serving.plugins.staticfile;

import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.nio.file.Path;
import java.nio.file.Paths;

/** Tests that a configured web root cannot select a directory outside the server home. */
public class WebRootTest {

    private static final Path HOME = Paths.get("/opt/djl").toAbsolutePath().normalize();

    @DataProvider(name = "containedRoots")
    public Object[][] containedRoots() {
        return new Object[][] {
            {"public", "public"},
            {"webapp/dist", "webapp/dist"},
            // Normalized rather than refused: it lands back inside the home.
            {"public/../public", "public"},
            {"./public", "public"},
        };
    }

    @Test(dataProvider = "containedRoots")
    public void testContainedWebRootIsUsed(String configured, String expected) {
        Assert.assertEquals(WebRoot.resolve(HOME, configured), HOME.resolve(expected).normalize());
    }

    @DataProvider(name = "escapingRoots")
    public Object[][] escapingRoots() {
        return new Object[][] {
            // A root outside the server home would make the handler's own containment checks
            // vacuous, since they compare against this root.
            {"../../../../"},
            {".."},
            {"../sibling"},
            {"public/../../etc"},
            // Absolute: Path.resolve returns it unchanged, so it has to be refused here.
            {"/etc"},
            {"/"},
            // Invalid rather than escaping, but must not propagate out of class init.
            {"pub\u0000lic"},
        };
    }

    @Test(dataProvider = "escapingRoots")
    public void testEscapingWebRootFallsBackToDefault(String configured) {
        Assert.assertEquals(WebRoot.resolve(HOME, configured), HOME.resolve(WebRoot.DEFAULT));
    }

    @Test
    public void testResultIsAlwaysInsideHome() {
        String[] values = {"public", "../../..", "/etc", "a/b/c", "..", "/", "pub\u0000lic"};
        for (String configured : values) {
            Path root = WebRoot.resolve(HOME, configured);
            Assert.assertTrue(root.startsWith(HOME), "web root escaped the server home: " + root);
        }
    }
}
