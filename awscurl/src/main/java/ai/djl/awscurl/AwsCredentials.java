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

import ai.djl.util.Ec2Utils;
import ai.djl.util.Utils;

import com.google.gson.annotations.SerializedName;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;

/** A class represents AWS credentials. */
public class AwsCredentials {

    public static final String ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY_ID";
    public static final String ALTERNATE_ACCESS_KEY_ENV_VAR = "AWS_ACCESS_KEY";
    public static final String SECRET_KEY_ENV_VAR = "AWS_SECRET_KEY";
    public static final String ALTERNATE_SECRET_KEY_ENV_VAR = "AWS_SECRET_ACCESS_KEY";
    public static final String AWS_SESSION_TOKEN_ENV_VAR = "AWS_SESSION_TOKEN";

    public static final String ACCESS_KEY_SYSTEM_PROPERTY = "aws.accessKeyId";
    public static final String SECRET_KEY_SYSTEM_PROPERTY = "aws.secretKey";
    public static final String SESSION_TOKEN_SYSTEM_PROPERTY = "aws.sessionToken";

    public static final String DEFAULT_PROFILE_NAME = "default";
    public static final String AWS_PROFILE_ENVIRONMENT_VARIABLE = "AWS_PROFILE";
    public static final String AWS_PROFILE_SYSTEM_PROPERTY = "aws.profile";

    @SerializedName("AccessKeyId")
    private String awsAccessKey;

    @SerializedName("SecretAccessKey")
    private String awsSecretKey;

    @SerializedName("Token")
    private String sessionToken;

    private String region;

    /**
     * Constructs a new {@code AWSCredentials} instance.
     *
     * @param awsAccessKey the access key
     * @param awsSecretKey the secret key
     * @param sessionToken the session token
     */
    public AwsCredentials(String awsAccessKey, String awsSecretKey, String sessionToken) {
        this(awsAccessKey, awsSecretKey, sessionToken, null);
    }

    /**
     * Constructs a new {@code AWSCredentials} instance.
     *
     * @param awsAccessKey the access key
     * @param awsSecretKey the secret key
     * @param sessionToken the session token
     * @param region the AWS region name
     */
    public AwsCredentials(
            String awsAccessKey, String awsSecretKey, String sessionToken, String region) {
        this.awsAccessKey = awsAccessKey.trim();
        this.awsSecretKey = awsSecretKey.trim();
        if (sessionToken != null) {
            this.sessionToken = sessionToken.trim();
        }
        if (region != null) {
            this.region = region.trim();
        }
    }

    /**
     * Returns the access key.
     *
     * @return the access key
     */
    public String getAWSAccessKeyId() {
        return awsAccessKey;
    }

    /**
     * Returns the secret key.
     *
     * @return the secret key
     */
    public String getAWSSecretKey() {
        return awsSecretKey;
    }

    /**
     * Returns the session token.
     *
     * @return the session token
     */
    public String getSessionToken() {
        return sessionToken;
    }

    /**
     * Returns the AWS region name.
     *
     * @return the AWS region name
     */
    public String getRegion() {
        return region;
    }

    /**
     * Returns the {@code AWSCredentials} from profile name.
     *
     * @param profile the profile name
     * @return the {@code AWSCredentials}
     */
    public static AwsCredentials getCredentials(String profile) {
        if (!StringUtils.isEmpty(profile)) {
            return loadFromProfile(profile);
        }

        String accessKey = Utils.getEnvOrSystemProperty(ACCESS_KEY_ENV_VAR);
        if (accessKey == null) {
            accessKey = Utils.getEnvOrSystemProperty(ALTERNATE_ACCESS_KEY_ENV_VAR);
        }
        String secretKey = Utils.getEnvOrSystemProperty(SECRET_KEY_ENV_VAR);
        if (secretKey == null) {
            secretKey = Utils.getEnvOrSystemProperty(ALTERNATE_SECRET_KEY_ENV_VAR);
        }
        String sessionToken = Utils.getEnvOrSystemProperty(AWS_SESSION_TOKEN_ENV_VAR);

        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AwsCredentials(accessKey, secretKey, sessionToken);
        }

        accessKey = System.getProperty(ACCESS_KEY_SYSTEM_PROPERTY);
        secretKey = System.getProperty(SECRET_KEY_SYSTEM_PROPERTY);
        sessionToken = System.getProperty(SESSION_TOKEN_SYSTEM_PROPERTY);
        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AwsCredentials(accessKey, secretKey, sessionToken);
        }

        String cred =
                Ec2Utils.readMetadata("identity-credentials/ec2/security-credentials/ec2-instance");
        if (cred != null && !cred.isEmpty()) {
            return JsonUtils.GSON.fromJson(cred, AwsCredentials.class);
        }

        return loadFromProfile(getDefaultProfileName());
    }

    private static AwsCredentials loadFromProfile(String profile) {
        String file = Utils.getEnvOrSystemProperty("AWS_SHARED_CREDENTIALS_FILE");
        if (file != null) {
            Path profileFile = Paths.get(file);
            if (Files.isRegularFile(profileFile)) {
                return loadProfileCredentials(profileFile, profile);
            }
            return null;
        }
        Path dir = Paths.get(System.getProperty("user.home")).resolve(".aws");
        Path profileFile = dir.resolve("credentials");
        if (Files.isRegularFile(profileFile)) {
            return loadProfileCredentials(profileFile, profile);
        }

        file = Utils.getEnvOrSystemProperty("AWS_CONFIG_FILE");
        if (file != null) {
            profileFile = Paths.get(file);
            if (Files.isRegularFile(profileFile)) {
                return loadProfileCredentials(profileFile, profile);
            }
            return null;
        }
        profileFile = dir.resolve("config");
        if (Files.isRegularFile(profileFile)) {
            return loadProfileCredentials(profileFile, profile);
        }
        return null;
    }

    private static String getDefaultProfileName() {
        String profileName = Utils.getEnvOrSystemProperty(AWS_PROFILE_ENVIRONMENT_VARIABLE);
        if (!StringUtils.isEmpty(profileName)) {
            return profileName;
        }

        profileName = System.getProperty(AWS_PROFILE_SYSTEM_PROPERTY);
        if (!StringUtils.isEmpty(profileName)) {
            return profileName;
        }

        return DEFAULT_PROFILE_NAME;
    }

    private static AwsCredentials loadProfileCredentials(Path file, String profile) {
        Map<String, String> map = loadProfile(file, profile);
        String accessKey = map.get("aws_access_key_id");
        String secretKey = map.get("aws_secret_access_key");
        String sessionToken = map.get("aws_session_token");
        String region = map.get("region");
        if (!StringUtils.isEmpty(accessKey) && !StringUtils.isEmpty(secretKey)) {
            return new AwsCredentials(accessKey, secretKey, sessionToken, region);
        }
        return null;
    }

    private static Map<String, String> loadProfile(Path file, String profile) {
        Map<String, String> map = new ConcurrentHashMap<>();
        try (Scanner scanner = new Scanner(file, StandardCharsets.UTF_8)) {
            boolean profileFound = false;
            while (scanner.hasNextLine()) {
                String line = scanner.nextLine().trim();
                if (line.isEmpty() || line.startsWith("#")) {
                    continue;
                }

                if (line.startsWith("[") && line.endsWith("]")) {
                    String profileName = line.substring(1, line.length() - 1);
                    if (profileName.startsWith("profile ")) {
                        profileName = profileName.substring("profile ".length()).trim();
                    }
                    if (profile.equalsIgnoreCase(profileName)) {
                        profileFound = true;
                    } else if (profileFound) {
                        return map;
                    }
                } else if (profileFound) {
                    String[] pair = line.split("=", 2);
                    if (pair.length != 2) {
                        throw new IllegalArgumentException(
                                "Invalid property format in the line: " + line);
                    }
                    map.put(pair[0].trim(), pair[1].trim());
                }
            }
        } catch (IOException e) {
            throw new AssertionError(e);
        }
        return map;
    }
}
