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

import ai.djl.engine.Engine;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.Platform;
import ai.djl.util.Utils;

import java.io.IOException;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

final class TokenUtils {

    private static HuggingFaceTokenizer tokenizer = getTokenizer();

    private TokenUtils() {}

    static int countTokens(List<? extends CharSequence> list) {
        int count = 0;
        for (CharSequence item : list) {
            if (tokenizer != null) {
                Encoding encoding = tokenizer.encode(item.toString());
                count += encoding.getIds().length;
            } else {
                String[] token = item.toString().split("\\s");
                count += token.length;
            }
        }
        return count;
    }

    static void setTokenizer() {
        tokenizer = getTokenizer();
    }

    @SuppressWarnings("PMD.SystemPrintln")
    private static HuggingFaceTokenizer getTokenizer() {
        try {
            Path cacheDir = Utils.getEngineCacheDir("tokenizers");
            Platform platform = Platform.detectPlatform("tokenizers");
            String classifier = platform.getClassifier();
            String version = platform.getVersion();
            Path dir = cacheDir.resolve(version + '-' + classifier);
            downloadLibs(dir, version, classifier, System.mapLibraryName("tokenizers"));
            if (classifier.startsWith("win-")) {
                downloadLibs(dir, version, classifier, "libwinpthread-1.dll");
                downloadLibs(dir, version, classifier, "libgcc_s_seh-1.dll");
                downloadLibs(dir, version, classifier, "libstdc++-6.dll");
            }
        } catch (IOException e) {
            throw new IllegalArgumentException("Failed to load HuggingFace tokenizer.", e);
        }

        HuggingFaceTokenizer.Builder builder = HuggingFaceTokenizer.builder();
        String name = Utils.getEnvOrSystemProperty("TOKENIZER");
        if (name != null) {
            Path path = Paths.get(name);
            if (Files.exists(path)) {
                builder.optTokenizerPath(path);
            } else {
                builder.optTokenizerName(name);
            }
            try {
                return builder.optAddSpecialTokens(false).build();
            } catch (Exception e) {
                AwsCurl.logger.warn("", e);
                System.out.println(
                        "Invalid tokenizer: "
                                + name
                                + ", please unset environment variable TOKENIZER if don't want to"
                                + " use tokenizer");
            }
        }
        return null;
    }

    private static void downloadLibs(Path dir, String version, String classifier, String libName)
            throws IOException {
        Path path = dir.resolve(libName);
        if (!Files.exists(path)) {
            Files.createDirectories(dir);
            String djlVersion = Engine.getDjlVersion().replaceAll("-SNAPSHOT", "");
            String encodedLibName;
            if (libName.contains("++")) {
                encodedLibName = libName.replaceAll("\\+\\+", "%2B%2B");
            } else {
                encodedLibName = libName;
            }
            String url =
                    "https://publish.djl.ai/tokenizers/"
                            + version.split("-")[0]
                            + "/jnilib/"
                            + djlVersion
                            + '/'
                            + classifier
                            + '/'
                            + encodedLibName;
            DownloadUtils.download(new URL(url), path, null);
        }
    }
}
