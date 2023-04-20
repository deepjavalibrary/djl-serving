/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

import ai.djl.serving.Arguments;
import ai.djl.serving.wlm.util.WlmConfigManager;
import ai.djl.util.Utils;

import io.netty.handler.ssl.SslContext;
import io.netty.handler.ssl.SslContextBuilder;
import io.netty.handler.ssl.util.SelfSignedCertificate;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.management.ManagementFactory;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.GeneralSecurityException;
import java.security.KeyException;
import java.security.KeyFactory;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.CertificateFactory;
import java.security.cert.X509Certificate;
import java.security.spec.InvalidKeySpecException;
import java.security.spec.PKCS8EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import java.util.Collection;
import java.util.Enumeration;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Properties;
import java.util.function.Consumer;

/** A class that hold configuration information. */
public final class ConfigManager {

    private static final String INFERENCE_ADDRESS = "inference_address";
    private static final String MANAGEMENT_ADDRESS = "management_address";
    private static final String LOAD_MODELS = "load_models";
    private static final String LOAD_WORKFLOWS = "load_workflows";
    private static final String WAIT_MODEL_LOADING = "wait_model_loading";
    private static final String ALLOW_MULTI_STATUS = "allow_multi_status";
    private static final String NUMBER_OF_NETTY_THREADS = "number_of_netty_threads";
    private static final String JOB_QUEUE_SIZE = "job_queue_size";
    private static final String MAX_IDLE_TIME = "max_idle_time";
    private static final String BATCH_SIZE = "batch_size";
    private static final String MAX_BATCH_DELAY = "max_batch_delay";
    private static final String RESERVED_MEMORY_MB = "reserved_memory_mb";
    private static final String CORS_ALLOWED_ORIGIN = "cors_allowed_origin";
    private static final String CORS_ALLOWED_METHODS = "cors_allowed_methods";
    private static final String CORS_ALLOWED_HEADERS = "cors_allowed_headers";
    private static final String KEYSTORE = "keystore";
    private static final String KEYSTORE_PASS = "keystore_pass";
    private static final String KEYSTORE_TYPE = "keystore_type";
    private static final String CERTIFICATE_FILE = "certificate_file";
    private static final String PRIVATE_KEY_FILE = "private_key_file";
    private static final String MAX_REQUEST_SIZE = "max_request_size";
    private static final String MODEL_STORE = "model_store";
    private static final String MODEL_URL_PATTERN = "model_url_pattern";
    private static final String LOAD_ON_DEVICES = "load_on_devices";
    private static final String PLUGIN_FOLDER = "plugin_folder";
    private static final String CHUNKED_READ_TIMEOUT = "chunked_read_timeout";

    // Configuration which are not documented or enabled through environment variables
    private static final String USE_NATIVE_IO = "use_native_io";
    private static final String IO_RATIO = "io_ratio";

    private static final int DEF_MAX_REQUEST_SIZE = 64 * 1024 * 1024;

    private static ConfigManager instance;

    private Properties prop;

    private ConfigManager(Arguments args) {
        prop = new Properties();

        Path file = args.getConfigFile();
        if (file != null) {
            try (InputStream stream = Files.newInputStream(file)) {
                prop.load(stream);
            } catch (IOException e) {
                throw new IllegalArgumentException("Unable to read configuration file", e);
            }
            prop.put("configFile", file.toString());
        }

        String modelStore = args.getModelStore();
        if (modelStore != null) {
            prop.setProperty(MODEL_STORE, modelStore);
        }

        String[] models = args.getModels();
        if (models != null) {
            prop.setProperty(LOAD_MODELS, String.join(",", models));
        }
        String[] workflows = args.getWorkflows();
        if (workflows != null) {
            prop.setProperty(LOAD_WORKFLOWS, String.join(",", workflows));
        }
        for (Map.Entry<String, String> env : Utils.getenv().entrySet()) {
            String key = env.getKey();
            if (key.startsWith("SERVING_")) {
                prop.put(key.substring(8).toLowerCase(Locale.ROOT), env.getValue());
            }
        }
    }

    /**
     * Initialize the global {@code ConfigManager} instance.
     *
     * @param args the command line arguments
     */
    public static void init(Arguments args) {
        instance = new ConfigManager(args);
        // set default system properties
        if (System.getProperty("ai.djl.pytorch.num_interop_threads") == null) {
            System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
        }
        if (System.getProperty("ai.djl.pytorch.num_threads") == null
                && Utils.getenv("OMP_NUM_THREADS") == null) {
            System.setProperty("ai.djl.pytorch.num_threads", "1");
        }
        if (System.getProperty("ai.djl.onnxruntime.num_interop_threads") == null) {
            System.setProperty("ai.djl.onnxruntime.num_interop_threads", "1");
        }
        if (System.getProperty("ai.djl.onnxruntime.num_threads") == null) {
            System.setProperty("ai.djl.onnxruntime.num_threads", "1");
        }
        if (System.getProperty("log4j2.contextSelector") == null) {
            // turn on async logging by default
            System.setProperty(
                    "log4j2.contextSelector",
                    "org.apache.logging.log4j.core.async.AsyncLoggerContextSelector");
        }

        // Disable alternative engine for Python in djl-serving
        if (System.getProperty("ai.djl.python.disable_alternative") == null) {
            System.setProperty("ai.djl.python.disable_alternative", "true");
        }

        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        instance.withIntProperty(JOB_QUEUE_SIZE, wlmc::setJobQueueSize);
        instance.withIntProperty(MAX_IDLE_TIME, wlmc::setMaxIdleSeconds);
        instance.withIntProperty(BATCH_SIZE, wlmc::setBatchSize);
        instance.withIntProperty(MAX_BATCH_DELAY, wlmc::setMaxBatchDelayMillis);
        instance.withIntProperty(RESERVED_MEMORY_MB, wlmc::setReservedMemoryMb);
        wlmc.setLoadOnDevices(instance.getLoadOnDevices());
    }

    /**
     * Returns the singleton {@code ConfigManager} instance.
     *
     * @return the singleton {@code ConfigManager} instance
     */
    public static ConfigManager getInstance() {
        return instance;
    }

    /**
     * Returns the models server socket connector.
     *
     * @param type the type of connector
     * @return the {@code Connector}
     */
    public Connector getConnector(Connector.ConnectorType type) {
        String binding;
        if (type == Connector.ConnectorType.MANAGEMENT) {
            binding = prop.getProperty(MANAGEMENT_ADDRESS, "http://127.0.0.1:8080");
        } else {
            binding = prop.getProperty(INFERENCE_ADDRESS, "http://127.0.0.1:8080");
        }
        return Connector.parse(binding, type);
    }

    /**
     * Returns the configured netty threads.
     *
     * @return the configured netty threads
     */
    public int getNettyThreads() {
        return getIntProperty(NUMBER_OF_NETTY_THREADS, 0);
    }

    /**
     * Returns the model server home directory.
     *
     * @return the model server home directory
     */
    public static String getModelServerHome() {
        String home = Utils.getenv("MODEL_SERVER_HOME");
        if (home == null) {
            home = System.getProperty("MODEL_SERVER_HOME");
            if (home == null) {
                home = getCanonicalPath(".");
                return home;
            }
        }

        Path dir = Paths.get(home);
        if (!Files.isDirectory(dir)) {
            throw new IllegalArgumentException("Model server home not exist: " + home);
        }
        home = getCanonicalPath(dir);
        return home;
    }

    /**
     * Returns if model server should wait for model initialization on startup.
     *
     * @return true if model server should wait for model initialization on startup
     */
    public boolean waitModelLoading() {
        return Boolean.parseBoolean(prop.getProperty(WAIT_MODEL_LOADING, "true"));
    }

    /**
     * Returns if allows return MULTI-STATUS HTTP code.
     *
     * @return true if allows return MULTI-STATUS HTTP code
     */
    public boolean allowsMultiStatus() {
        return Boolean.parseBoolean(prop.getProperty(ALLOW_MULTI_STATUS));
    }

    /**
     * Returns the model store location.
     *
     * @return the model store location
     */
    public Path getModelStore() {
        return getPathProperty(MODEL_STORE);
    }

    /**
     * Returns the allowed model url pattern regex.
     *
     * @return the allowed model url pattern regex
     */
    public String getModelUrlPattern() {
        return prop.getProperty(MODEL_URL_PATTERN);
    }

    /**
     * Returns the model urls that to be loaded at startup.
     *
     * @return the model urls that to be loaded at startup
     */
    public String getLoadModels() {
        return prop.getProperty(LOAD_MODELS);
    }

    /**
     * Returns the workflow urls that to be loaded at startup.
     *
     * @return the workflow urls that to be loaded at startup
     */
    public String getLoadWorkflows() {
        return prop.getProperty(LOAD_WORKFLOWS);
    }

    /**
     * Returns the devices the default model will be loaded on at startup.
     *
     * @return the devices the default model will be loaded on at startup
     */
    public String getLoadOnDevices() {
        return prop.getProperty(LOAD_ON_DEVICES, "*");
    }

    /**
     * Returns the CORS allowed origin setting.
     *
     * @return the CORS allowed origin setting
     */
    public String getCorsAllowedOrigin() {
        return prop.getProperty(CORS_ALLOWED_ORIGIN);
    }

    /**
     * Returns the CORS allowed method setting.
     *
     * @return the CORS allowed method setting
     */
    public String getCorsAllowedMethods() {
        return prop.getProperty(CORS_ALLOWED_METHODS);
    }

    /**
     * Returns the CORS allowed headers setting.
     *
     * @return the CORS allowed headers setting
     */
    public String getCorsAllowedHeaders() {
        return prop.getProperty(CORS_ALLOWED_HEADERS);
    }

    /**
     * return the folder where the model search for plugins.
     *
     * @return the configured plugin folder or the default folder.
     */
    public Path getPluginFolder() {
        return getPathProperty(PLUGIN_FOLDER, "plugins");
    }

    /**
     * Returns a {@code SSLContext} instance.
     *
     * @return a {@code SSLContext} instance
     * @throws IOException if failed to read certificate file
     * @throws GeneralSecurityException if failed to initialize {@code SSLContext}
     */
    public SslContext getSslContext() throws IOException, GeneralSecurityException {
        List<String> supportedCiphers =
                Arrays.asList(
                        "TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA",
                        "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256");

        PrivateKey privateKey;
        X509Certificate[] chain;
        Path keyStoreFile = getPathProperty(KEYSTORE);
        Path privateKeyFile = getPathProperty(PRIVATE_KEY_FILE);
        Path certificateFile = getPathProperty(CERTIFICATE_FILE);
        if (keyStoreFile != null) {
            char[] keystorePass = getProperty(KEYSTORE_PASS, "changeit").toCharArray();
            String keystoreType = getProperty(KEYSTORE_TYPE, "PKCS12");
            KeyStore keyStore = KeyStore.getInstance(keystoreType);
            try (InputStream is = Files.newInputStream(keyStoreFile)) {
                keyStore.load(is, keystorePass);
            }

            Enumeration<String> en = keyStore.aliases();
            String keyAlias = null;
            while (en.hasMoreElements()) {
                String alias = en.nextElement();
                if (keyStore.isKeyEntry(alias)) {
                    keyAlias = alias;
                    break;
                }
            }

            if (keyAlias == null) {
                throw new KeyException("No key entry found in keystore.");
            }

            privateKey = (PrivateKey) keyStore.getKey(keyAlias, keystorePass);

            Certificate[] certs = keyStore.getCertificateChain(keyAlias);
            chain = new X509Certificate[certs.length];
            for (int i = 0; i < certs.length; ++i) {
                chain[i] = (X509Certificate) certs[i];
            }
        } else if (privateKeyFile != null && certificateFile != null) {
            privateKey = loadPrivateKey(privateKeyFile);
            chain = loadCertificateChain(certificateFile);
        } else {
            SelfSignedCertificate ssc = new SelfSignedCertificate();
            privateKey = ssc.key();
            chain = new X509Certificate[] {ssc.cert()};
        }

        return SslContextBuilder.forServer(privateKey, chain)
                .protocols("TLSv1.2")
                .ciphers(supportedCiphers)
                .build();
    }

    /**
     * Returns the ChunkedBytesSupplier read time in minutes.
     *
     * @return the ChunkedBytesSupplier read time in minutes
     */
    public int getChunkedReadTimeout() {
        return getIntProperty(CHUNKED_READ_TIMEOUT, 2);
    }

    /**
     * Returns the value with the specified key in this configuration.
     *
     * @param key the key
     * @param def a default value
     * @return the value with the specified key in this configuration
     */
    public String getProperty(String key, String def) {
        return prop.getProperty(key, def);
    }

    /**
     * Prints out this configuration.
     *
     * @return a string representation of this configuration
     */
    public String dumpConfigurations() {
        WlmConfigManager wlmc = WlmConfigManager.getInstance();
        Runtime runtime = Runtime.getRuntime();
        return "\nModel server home: "
                + getModelServerHome()
                + "\nCurrent directory: "
                + getCanonicalPath(".")
                + "\nTemp directory: "
                + System.getProperty("java.io.tmpdir")
                + "\nCommand line: "
                + String.join(" ", ManagementFactory.getRuntimeMXBean().getInputArguments())
                + "\nNumber of CPUs: "
                + runtime.availableProcessors()
                + "\nMax heap size: "
                + (runtime.maxMemory() / 1024 / 1024)
                + "\nConfig file: "
                + prop.getProperty("configFile", "N/A")
                + "\nInference address: "
                + getConnector(Connector.ConnectorType.INFERENCE)
                + "\nManagement address: "
                + getConnector(Connector.ConnectorType.MANAGEMENT)
                + "\nDefault job_queue_size: "
                + wlmc.getJobQueueSize()
                + "\nDefault batch_size: "
                + wlmc.getBatchSize()
                + "\nDefault max_batch_delay: "
                + wlmc.getMaxBatchDelayMillis()
                + "\nDefault max_idle_time: "
                + wlmc.getMaxIdleSeconds()
                + "\nModel Store: "
                + (getModelStore() == null ? "N/A" : getModelStore())
                + "\nInitial Models: "
                + (getLoadModels() == null ? "N/A" : getLoadModels())
                + "\nInitial Workflows: "
                + (getLoadWorkflows() == null ? "N/A" : getLoadWorkflows())
                + "\nNetty threads: "
                + getNettyThreads()
                + "\nMaximum Request Size: "
                + prop.getProperty(MAX_REQUEST_SIZE, String.valueOf(getMaxRequestSize()));
    }

    /**
     * Returns if use netty native IO.
     *
     * @return {@code true} if use netty native IO
     */
    public boolean useNativeIo() {
        return Boolean.parseBoolean(prop.getProperty(USE_NATIVE_IO, "true"));
    }

    /**
     * Returns the native IO ratio.
     *
     * @return the native IO ratio
     */
    public int getIoRatio() {
        return getIntProperty(IO_RATIO, 50);
    }

    /**
     * Returns the maximum allowed request size in bytes.
     *
     * @return the maximum allowed request size in bytes
     */
    public int getMaxRequestSize() {
        return getIntProperty(MAX_REQUEST_SIZE, DEF_MAX_REQUEST_SIZE);
    }

    private int getIntProperty(String key, int def) {
        String value = prop.getProperty(key);
        if (value == null) {
            return def;
        }
        return Integer.parseInt(value);
    }

    private void withIntProperty(String key, Consumer<Integer> f) {
        if (prop.containsKey(key)) {
            f.accept(Integer.parseInt(prop.getProperty(key)));
        }
    }

    private Path getPathProperty(String key) {
        return getPathProperty(key, null);
    }

    private Path getPathProperty(String key, String defaultValue) {
        String property = prop.getProperty(key, defaultValue);
        if (property == null) {
            return null;
        }
        Path path = Paths.get(property);
        if (!path.isAbsolute()) {
            path = Paths.get(getModelServerHome()).resolve(path);
        }
        return path;
    }

    private static String getCanonicalPath(Path file) {
        try {
            return file.toRealPath().toString();
        } catch (IOException e) {
            return file.toAbsolutePath().toString();
        }
    }

    private static String getCanonicalPath(String path) {
        if (path == null) {
            return null;
        }
        return getCanonicalPath(Paths.get(path));
    }

    private PrivateKey loadPrivateKey(Path keyFile) throws IOException, GeneralSecurityException {
        KeyFactory keyFactory = KeyFactory.getInstance("RSA");
        try (ByteArrayOutputStream os = new ByteArrayOutputStream()) {
            Files.copy(keyFile, os);
            String content = os.toString(StandardCharsets.UTF_8);
            content = content.replaceAll("-----(BEGIN|END)( RSA)? PRIVATE KEY-----\\s*", "");
            byte[] buf = Base64.getMimeDecoder().decode(content);
            try {
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            } catch (InvalidKeySpecException e) {
                // old private key is OpenSSL format private key
                buf = OpenSslKey.convertPrivateKey(buf);
                PKCS8EncodedKeySpec privKeySpec = new PKCS8EncodedKeySpec(buf);
                return keyFactory.generatePrivate(privKeySpec);
            }
        }
    }

    private X509Certificate[] loadCertificateChain(Path keyFile)
            throws IOException, GeneralSecurityException {
        CertificateFactory cf = CertificateFactory.getInstance("X.509");
        try (InputStream is = Files.newInputStream(keyFile)) {
            Collection<? extends Certificate> certs = cf.generateCertificates(is);
            int i = 0;
            X509Certificate[] chain = new X509Certificate[certs.size()];
            for (Certificate cert : certs) {
                chain[i++] = (X509Certificate) cert;
            }
            return chain;
        }
    }
}
