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

import java.net.URL;
import java.net.URLClassLoader;
import java.security.AccessController;
import java.security.PrivilegedAction;

/** A {@code URLClassLoader} that can add new class at runtime. */
public class MutableClassLoader extends URLClassLoader {

    private static final MutableClassLoader INSTANCE =
            AccessController.doPrivileged(
                    (PrivilegedAction<MutableClassLoader>) MutableClassLoader::new);

    /**
     * Constructs a new URLClassLoader for the given URLs. The URLs will be searched in the order
     * specified for classes and resources after first searching in the specified parent class
     * loader. Any URL that ends with a '/' is assumed to refer to a directory. Otherwise, the URL
     * is assumed to refer to a JAR file which will be downloaded and opened as needed.
     *
     * <p>If there is a security manager, this method first calls the security manager's {@code
     * checkCreateClassLoader} method to ensure creation of a class loader is allowed.
     *
     * @throws SecurityException if a security manager exists and its {@code checkCreateClassLoader}
     *     method doesn't allow creation of a class loader.
     * @throws NullPointerException if {@code urls} is {@code null}.
     * @see SecurityManager#checkCreateClassLoader
     */
    public MutableClassLoader() {
        super(new URL[0]);
    }

    /**
     * Returns the singleton instance of {@code ServingClassLoader}.
     *
     * @return the singleton instance of {@code ServingClassLoader}
     */
    public static MutableClassLoader getInstance() {
        return INSTANCE;
    }

    /** {@inheritDoc} */
    @Override
    public void addURL(URL url) {
        super.addURL(url);
    }
}
