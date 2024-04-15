/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.awscurl;

import ai.djl.util.Utils;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class AwsCredentialsTest {

    @Test
    public void testFromProfile() throws IOException {
        String userHome = System.getProperty("user.home");
        Path dir = Paths.get("build/.aws");
        try {
            System.setProperty("user.home", "build");
            AwsCredentials credentials = AwsCredentials.getCredentials(null);
            if (credentials != null) {
                // got credentials from instance profile
                Assert.assertNotNull(credentials.getAWSAccessKeyId());
            }

            Files.createDirectories(dir);
            Path config = dir.resolve("config");
            try (BufferedWriter writer = Files.newBufferedWriter(config)) {
                writer.write("# comments\r\n");
                writer.write("[default]\n");
                writer.write("aws_access_key_id=ACCESS_KEY_ID\n");
                writer.write("aws_secret_access_key=AWS_SECRET_ACCESS_KEY\n");
                writer.write("aws_session_token=AWS_SESSION_TOKEN\n");
                writer.write("region = us-east-1\n\n");
            }

            credentials = AwsCredentials.getCredentials(null);
            Assert.assertNotNull(credentials);

            credentials = AwsCredentials.getCredentials("default");
            Assert.assertNotNull(credentials);
            Assert.assertEquals(credentials.getRegion(), "us-east-1");

            System.setProperty("AWS_CONFIG_FILE", config.toAbsolutePath().toString());
            System.setProperty(AwsCredentials.AWS_PROFILE_SYSTEM_PROPERTY, "default");
            credentials = AwsCredentials.getCredentials("default");
            Assert.assertNotNull(credentials);

            System.setProperty("AWS_CONFIG_FILE", config.resolveSibling("non-exist").toString());
            System.setProperty(AwsCredentials.AWS_PROFILE_ENVIRONMENT_VARIABLE, "default");
            credentials = AwsCredentials.getCredentials("default");
            Assert.assertNull(credentials);

            Path credFile = dir.resolve("credentials");
            try (BufferedWriter writer = Files.newBufferedWriter(credFile)) {
                writer.write("[profile default]\n");
                writer.write("aws_access_key_id=ACCESS_KEY_ID\n");
                writer.write("aws_secret_access_key=AWS_SECRET_ACCESS_KEY\n");
            }
            credentials = AwsCredentials.getCredentials("default");
            Assert.assertNull(credentials.getSessionToken());

            System.setProperty("AWS_SHARED_CREDENTIALS_FILE", credFile.toAbsolutePath().toString());
            credentials = AwsCredentials.getCredentials("default");
            Assert.assertNull(credentials.getSessionToken());

            System.setProperty(
                    "AWS_SHARED_CREDENTIALS_FILE", config.resolveSibling("non-exist").toString());
            credentials = AwsCredentials.getCredentials(null);
            if (credentials != null) {
                // got credentials from instance profile
                Assert.assertNotNull(credentials.getAWSAccessKeyId());
            }

            credentials = AwsCredentials.getCredentials("non-exist");
            Assert.assertNull(credentials);
        } finally {
            System.setProperty("user.home", userHome);
            Utils.deleteQuietly(dir);
            System.clearProperty("AWS_CONFIG_FILE");
            System.clearProperty(AwsCredentials.AWS_PROFILE_SYSTEM_PROPERTY);
            System.clearProperty(AwsCredentials.AWS_PROFILE_ENVIRONMENT_VARIABLE);
        }
    }

    @Test
    public void testEnv() {
        if (Utils.getenv("AWS_ACCESS_KEY_ID") != null) {
            // Skip test if real credential is set
            return;
        }

        try {
            System.setProperty(AwsCredentials.ACCESS_KEY_SYSTEM_PROPERTY, "id");
            System.setProperty(AwsCredentials.SECRET_KEY_SYSTEM_PROPERTY, "key");
            System.setProperty(AwsCredentials.SESSION_TOKEN_SYSTEM_PROPERTY, "token");
            AwsCredentials credentials = AwsCredentials.getCredentials(null);
            Assert.assertNotNull(credentials.getSessionToken());

            System.setProperty(AwsCredentials.ALTERNATE_ACCESS_KEY_ENV_VAR, "id");
            System.setProperty(AwsCredentials.ALTERNATE_SECRET_KEY_ENV_VAR, "key");
            System.setProperty(AwsCredentials.AWS_SESSION_TOKEN_ENV_VAR, "token1");
            credentials = AwsCredentials.getCredentials(null);
            Assert.assertEquals(credentials.getSessionToken(), "token1");
        } finally {
            System.clearProperty(AwsCredentials.ACCESS_KEY_SYSTEM_PROPERTY);
            System.clearProperty(AwsCredentials.SECRET_KEY_SYSTEM_PROPERTY);
            System.clearProperty(AwsCredentials.SESSION_TOKEN_SYSTEM_PROPERTY);
            System.clearProperty(AwsCredentials.ALTERNATE_ACCESS_KEY_ENV_VAR);
            System.clearProperty(AwsCredentials.ALTERNATE_SECRET_KEY_ENV_VAR);
            System.clearProperty(AwsCredentials.AWS_SESSION_TOKEN_ENV_VAR);
        }
    }
}
