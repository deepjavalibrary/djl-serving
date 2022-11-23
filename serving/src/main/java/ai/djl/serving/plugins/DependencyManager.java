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
package ai.djl.serving.plugins;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineProvider;
import ai.djl.repository.zoo.ModelZoo;
import ai.djl.repository.zoo.ZooProvider;
import ai.djl.serving.util.ConfigManager;
import ai.djl.serving.util.MutableClassLoader;
import ai.djl.util.Utils;
import ai.djl.util.cuda.CudaUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Document;
import org.w3c.dom.Element;
import org.w3c.dom.NodeList;
import org.xml.sax.SAXException;

import java.io.IOException;
import java.io.InputStream;
import java.net.MalformedURLException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ServiceLoader;
import java.util.stream.Stream;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;

/** {@code DependencyManager} is responsible to manage extra maven dependencies. */
public class DependencyManager {

    private static final Logger logger = LoggerFactory.getLogger(DependencyManager.class);

    private static final DependencyManager INSTANCE = new DependencyManager();
    private static final String OSS_URL =
            "https://oss.sonatype.org/service/local/repositories/snapshots/content/";

    private Path depDir;

    DependencyManager() {
        String serverHome = ConfigManager.getModelServerHome();
        depDir = Paths.get(serverHome, "deps");
        if (Files.isDirectory(depDir)) {
            MutableClassLoader mc = MutableClassLoader.getInstance();
            try (Stream<Path> stream = Files.list(depDir)) {
                stream.forEach(
                        p -> {
                            if (p.toString().endsWith(".jar")) {
                                try {
                                    mc.addURL(p.toUri().toURL());
                                } catch (MalformedURLException e) {
                                    logger.warn("Invalid file system path: " + p, e);
                                }
                            }
                        });
            } catch (IOException e) {
                logger.warn("Failed to load dependencies from deps folder.", e);
            }
        }
    }

    /**
     * Returns the singleton instance of {@code DependencyManager}.
     *
     * @return the singleton instance of {@code DependencyManager}
     */
    public static DependencyManager getInstance() {
        return INSTANCE;
    }

    /**
     * Installs the engine dependencies if needed.
     *
     * @param engineName the engine name
     * @throws IOException if failed to download the dependency
     */
    public void installEngine(String engineName) throws IOException {
        if (Engine.hasEngine(engineName)) {
            return;
        }

        String djlVersion = Engine.getDjlVersion();
        switch (engineName) {
            case "OnnxRuntime":
                installDependency("ai.djl.onnxruntime:onnxruntime-engine:" + djlVersion);
                String ortVersion = getOrtVersion(djlVersion);
                if (CudaUtils.hasCuda()) {
                    installDependency("com.microsoft.onnxruntime:onnxruntime_gpu:" + ortVersion);
                } else {
                    installDependency("com.microsoft.onnxruntime:onnxruntime:" + ortVersion);
                }
                break;
            case "PaddlePaddle":
                installDependency("ai.djl.paddlepaddle:paddlepaddle-engine:" + djlVersion);
                break;
            case "TFLite":
                installDependency("ai.djl.tflite:tflite-engine:" + djlVersion);
                break;
            case "XGBoost":
                installDependency("ai.djl.ml.xgboost:xgboost:" + djlVersion);
                break;
            case "DLR":
                installDependency("ai.djl.dlr:dlr-engine:" + djlVersion);
                break;
            default:
                break;
        }

        // refresh EngineProvider
        MutableClassLoader mcl = MutableClassLoader.getInstance();
        for (EngineProvider provider : ServiceLoader.load(EngineProvider.class, mcl)) {
            Engine.registerEngine(provider);
        }

        // refresh ZooProvider
        for (ZooProvider provider : ServiceLoader.load(ZooProvider.class, mcl)) {
            ModelZoo.registerModelZoo(provider);
        }
    }

    /**
     * Installs the maven dependency.
     *
     * @param dependency the maven dependency
     * @throws IOException if failed to download the dependency
     */
    public synchronized void installDependency(String dependency) throws IOException {
        String[] tokens = dependency.split(":");
        if (tokens.length < 3) {
            throw new IllegalArgumentException("Invalid dependency: " + dependency);
        }
        Files.createDirectories(depDir);

        logger.info("Loading dependency: {}", dependency);
        String groupId = tokens[0].replace('.', '/');
        String artifactId = tokens[1];
        String version = tokens[2];
        String name;
        if (tokens.length == 3) {
            name = artifactId + '-' + version + ".jar";
        } else {
            name = artifactId + '-' + version + '-' + tokens[3] + ".jar";
        }

        Path file = depDir.resolve(name);
        if (Files.isRegularFile(file)) {
            logger.info("Found existing dependency: {}", name);
        } else {
            String link;
            if (version.endsWith("-SNAPSHOT")) {
                link = getSnapshotUrl(groupId, artifactId, version) + ".jar";
            } else {
                String maven = "https://search.maven.org/remotecontent?filepath=";
                link = maven + groupId + '/' + artifactId + '/' + version + '/' + name;
            }
            logger.info("Downloading dependency: {}.", link);
            Path tmp = depDir.resolve(name + ".tmp");
            try (InputStream is = Utils.openUrl(link)) {
                Files.copy(is, tmp);
                Utils.moveQuietly(tmp, file);
            } finally {
                Files.deleteIfExists(tmp);
            }
        }
        MutableClassLoader mcl = MutableClassLoader.getInstance();
        mcl.addURL(file.toUri().toURL());
    }

    private static String getSnapshotUrl(String groupId, String artifactId, String version)
            throws IOException {
        String groupPath = groupId.replace('.', '/');
        String url = OSS_URL + groupPath + '/' + artifactId + '/' + version + '/';
        try (InputStream is = Utils.openUrl(url + "maven-metadata.xml")) {
            Document doc = parseXml(is);
            NodeList nl = doc.getElementsByTagName("snapshot");
            Element element = (Element) nl.item(0);
            String timestamp = getElementValue(element, "timestamp");
            String buildNumber = getElementValue(element, "buildNumber");
            String v = version.substring(0, version.length() - 9);
            return url + artifactId + '-' + v + '-' + timestamp + '-' + buildNumber;
        }
    }

    private static Document parseXml(InputStream is) throws IOException {
        try {
            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            dbf.setFeature("http://apache.org/xml/features/disallow-doctype-decl", true);
            dbf.setXIncludeAware(false);
            dbf.setExpandEntityReferences(false);
            DocumentBuilder db = dbf.newDocumentBuilder();
            return db.parse(is);
        } catch (SAXException | ParserConfigurationException e) {
            throw new AssertionError("Failed to parse maven metadata", e);
        }
    }

    private String getOrtVersion(String djlVersion) throws IOException {
        String pom;
        if (djlVersion.endsWith("-SNAPSHOT")) {
            String groupId = "ai.djl.onnxruntime";
            pom = getSnapshotUrl(groupId, "onnxruntime-engine", djlVersion) + ".pom";
        } else {
            pom =
                    "https://search.maven.org/remotecontent?filepath=ai/djl/onnxruntime/onnxruntime-engine/"
                            + djlVersion
                            + "/onnxruntime-engine-"
                            + djlVersion
                            + ".pom";
        }
        try (InputStream is = Utils.openUrl(pom)) {
            Document doc = parseXml(is);
            NodeList nl = doc.getElementsByTagName("dependency");
            int len = nl.getLength();
            for (int i = 0; i < len; ++i) {
                Element element = (Element) nl.item(i);
                String group = getElementValue(element, "groupId");
                if ("com.microsoft.onnxruntime".equals(group)) {
                    return getElementValue(element, "version");
                }
            }
        }
        throw new AssertionError("Failed to find onnxruntime version.");
    }

    private static String getElementValue(Element element, String name) {
        NodeList nl = element.getElementsByTagName(name);
        Element node = (Element) nl.item(0);
        return node.getChildNodes().item(0).getTextContent();
    }
}
