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
package ai.djl.serving;

import ai.djl.repository.Repository;
import ai.djl.repository.RepositoryFactory;
import ai.djl.repository.SimpleRepository;

import java.net.URI;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Collections;
import java.util.Set;

class TempLocalRepositoryFactory implements RepositoryFactory {

    /** {@inheritDoc} */
    @Override
    public Repository newInstance(String name, URI uri) {
        Path path = Paths.get(parseFilePath(uri));
        return new MySimpleRepository(name, uri, path);
    }

    /** {@inheritDoc} */
    @Override
    public Set<String> getSupportedScheme() {
        return Collections.singleton("file");
    }

    private String parseFilePath(URI uri) {
        String uriPath = uri.getPath();
        if (uriPath == null) {
            uriPath = uri.getSchemeSpecificPart();
        }
        if (uriPath.startsWith("file:")) {
            // handle jar:file:/ url
            uriPath = uriPath.substring(5);
        }
        if (uriPath.startsWith("/") && System.getProperty("os.name").startsWith("Win")) {
            uriPath = uriPath.substring(1);
        }
        return uriPath;
    }

    private static final class MySimpleRepository extends SimpleRepository {

        public MySimpleRepository(String name, URI uri, Path path) {
            super(name, uri, path);
        }
    }
}
