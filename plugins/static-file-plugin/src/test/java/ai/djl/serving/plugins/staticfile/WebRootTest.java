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
import org.testng.SkipException;
import org.testng.annotations.AfterMethod;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

import java.io.IOException;
import java.nio.file.FileSystemException;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;

/** Tests that a configured web root cannot select a directory outside the server home. */
public class WebRootTest {

    private static final Path HOME = Paths.get("/opt/djl").toAbsolutePath().normalize();

    private Path tmp;

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
            // Absolute values are refused whatever they point at. One equal to the home would
            // otherwise pass containment and serve conf, logs and deps; one below it is
            // expressible relatively.
            {"/etc"},
            {"/"},
            {"/opt/djl"},
            {"/opt/djl/webapp"},
            {"/opt/djl/../djl/webapp"},
            // Relative, but normalize to the home itself, which is the same exposure.
            {"."},
            {""},
            {"public/.."},
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
        String[] values = {
            "public",
            "../../..",
            "/etc",
            "a/b/c",
            "..",
            "/",
            "pub\u0000lic",
            "/opt/djl",
            ".",
            "",
            "public/..",
            "/opt/djl/webapp"
        };
        for (String configured : values) {
            Path root = WebRoot.resolve(HOME, configured);
            Assert.assertTrue(root.startsWith(HOME), "web root escaped the server home: " + root);
            Assert.assertNotEquals(root, HOME, "web root selected the server home: " + configured);
        }
    }

    @BeforeMethod
    public void createTmp() throws IOException {
        tmp = Files.createTempDirectory("webroot-test");
    }

    @AfterMethod
    public void deleteTmp() throws IOException {
        if (tmp == null) {
            return;
        }
        // Does not follow links by default, so an outward-pointing link is unlinked, not walked.
        Files.walkFileTree(
                tmp,
                new SimpleFileVisitor<Path>() {
                    /** {@inheritDoc} */
                    @Override
                    public FileVisitResult visitFile(Path f, BasicFileAttributes attrs)
                            throws IOException {
                        Files.delete(f);
                        return FileVisitResult.CONTINUE;
                    }

                    /** {@inheritDoc} */
                    @Override
                    public FileVisitResult postVisitDirectory(Path d, IOException e)
                            throws IOException {
                        Files.delete(d);
                        return FileVisitResult.CONTINUE;
                    }
                });
    }

    /**
     * Creates a symbolic link, or skips the test where the platform will not allow one.
     *
     * <p>Windows refuses without a privilege the CI runner does not hold, and these cases are about
     * link following, so there is nothing left to assert without one.
     *
     * @param link the link to create
     * @param target what it points at
     * @return the created link
     * @throws IOException if creating the link fails for any other reason
     */
    private static Path symlinkOrSkip(Path link, Path target) throws IOException {
        try {
            return Files.createSymbolicLink(link, target);
        } catch (UnsupportedOperationException | FileSystemException e) {
            throw new SkipException("Symbolic links unavailable on this platform: " + e);
        }
    }

    @Test
    public void testRealRootInsideHomeIsReturned() throws IOException {
        Path home = Files.createDirectory(tmp.resolve("home"));
        Path root = Files.createDirectory(home.resolve("public"));
        Assert.assertEquals(WebRoot.realRootIn(home, root), root.toRealPath());
    }

    @Test
    public void testHomeReachedThroughALinkStillAcceptsAContainedRoot() throws IOException {
        // The should-pass cell: /opt -> /mnt/opt is ordinary, and must not be refused.
        Path real = Files.createDirectory(tmp.resolve("real-home"));
        Path root = Files.createDirectory(real.resolve("public"));
        Path home = symlinkOrSkip(tmp.resolve("home"), real);
        Assert.assertEquals(WebRoot.realRootIn(home, home.resolve("public")), root.toRealPath());
    }

    @Test(expectedExceptions = IOException.class)
    public void testRootSymlinkPointingOutsideIsRefused() throws IOException {
        Path home = Files.createDirectory(tmp.resolve("home"));
        Path outside = Files.createDirectory(tmp.resolve("outside"));
        Path root = symlinkOrSkip(home.resolve("public"), outside);
        WebRoot.realRootIn(home, root);
    }

    @Test(expectedExceptions = IOException.class)
    public void testIntermediateSymlinkPointingOutsideIsRefused() throws IOException {
        Path home = Files.createDirectory(tmp.resolve("home"));
        Path outside = Files.createDirectory(tmp.resolve("outside"));
        Files.createDirectory(outside.resolve("public"));
        symlinkOrSkip(home.resolve("web"), outside);
        WebRoot.realRootIn(home, home.resolve("web/public"));
    }

    @Test(expectedExceptions = IOException.class)
    public void testRootThatDoesNotExistIsRefused() throws IOException {
        Path home = Files.createDirectory(tmp.resolve("home"));
        WebRoot.realRootIn(home, home.resolve("public"));
    }
}
