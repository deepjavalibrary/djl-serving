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

import ai.djl.repository.FilenameUtils;
import ai.djl.util.RandomUtils;
import ai.djl.util.Utils;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.http.Header;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.entity.ContentType;
import org.apache.http.entity.mime.MultipartEntityBuilder;
import org.slf4j.ILoggerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/** The utility class to run curl like command. */
@SuppressWarnings("PMD.SystemPrintln")
public final class AwsCurl {

    static {
        String logLevel = System.getProperty("org.slf4j.simpleLogger.defaultLogLevel");
        if (logLevel == null) {
            System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "warn");
        }
        System.setProperty("org.slf4j.simpleLogger.showThreadName", "false");
        System.setProperty("org.slf4j.simpleLogger.showLogName", "false");
    }

    static Logger logger = LoggerFactory.getLogger(AwsCurl.class); // NOPMD

    private static final String SM_CUSTOM_HEADER = "X-Amzn-SageMaker-Custom-Attributes";

    private AwsCurl() {}

    /**
     * The main entrypoint.
     *
     * @param args the command line arguments.
     */
    @SuppressWarnings("PMD.AvoidAccessibilityAlteration")
    public static void main(String[] args) {
        run(args);
    }

    @SuppressWarnings("PMD.AvoidAccessibilityAlteration")
    static Result run(String[] args) {
        Result ret = new Result();
        String jarName = getJarName();
        Options options = Config.getOptions();
        DefaultParser parser = new DefaultParser();
        try {
            if (args.length == 0
                    || "-h".equalsIgnoreCase(args[0])
                    || "--help".equalsIgnoreCase(args[0])) {
                printHelp(jarName, options);
                return ret;
            }

            CommandLine cmd = parser.parse(options, args, null, false);
            List<String> cmdArgs = cmd.getArgList();
            if (cmdArgs.isEmpty()) {
                System.out.println("Missing URL in command line.");
                printHelp(jarName, options);
                ret.setHasError();
                return ret;
            }
            Config config = new Config(cmd);
            if (config.isVerbose()) {
                System.setProperty("org.slf4j.simpleLogger.defaultLogLevel", "info");
                try {
                    ILoggerFactory factory = LoggerFactory.getILoggerFactory();
                    Method method = factory.getClass().getDeclaredMethod("reset");
                    method.setAccessible(true);
                    method.invoke(factory);
                    Field field = logger.getClass().getDeclaredField("CONFIG_PARAMS");
                    field.setAccessible(true);
                    Object params = field.get(logger);
                    Method m = params.getClass().getDeclaredMethod("init");
                    m.setAccessible(true);
                    m.invoke(params);
                    logger = LoggerFactory.getLogger(AwsCurl.class);
                } catch (ReflectiveOperationException e) {
                    logger.debug(null, e);
                }
            }
            if (config.getError() != null) {
                System.out.println(config.getError());
                ret.setHasError();
                return ret;
            }

            String url = config.getUrl(cmdArgs.get(0));
            URI uri;
            try {
                uri = new URI(url);
                String scheme = uri.getScheme();
                if (scheme == null || !scheme.startsWith("http")) {
                    System.err.println("Only HTTP url is supported: " + url);
                    ret.setHasError();
                    return ret;
                }
            } catch (URISyntaxException e) {
                System.err.println("Invalid url: " + url);
                ret.setHasError();
                return ret;
            }

            String serviceName = config.getServiceName();
            AwsV4Signer signer;
            if (serviceName != null) {
                AwsCredentials credentials = AwsCredentials.getCredentials(config.getProfile());
                if (credentials == null) {
                    System.err.println("Could not load AWSCredentials.");
                    ret.setHasError();
                    return ret;
                }
                if (StringUtils.isEmpty(credentials.getAWSAccessKeyId())) {
                    System.err.println("Anonymous credentials is not supported.");
                    ret.setHasError();
                    return ret;
                }
                String region = config.getRegion();
                if (region == null) {
                    region = inferRegion(url, serviceName);
                    if (region == null) {
                        region = credentials.getRegion();
                        if (region == null) {
                            System.err.println("Not able to obtain region name from profile.");
                            ret.setHasError();
                            return ret;
                        }
                    }
                }
                signer = new AwsV4Signer(serviceName, region, credentials);
            } else {
                signer = null;
            }

            boolean insecure = config.isInsecure();
            boolean printHeader = config.isInclude() || config.isVerbose();
            int clients = config.getClients();
            int nRequests = config.getNumberOfRequests();

            AtomicInteger totalReq = new AtomicInteger(clients * nRequests);
            final List<Long> success = Collections.synchronizedList(new ArrayList<>());
            final List<Long> firstTokens = Collections.synchronizedList(new ArrayList<>());
            final AtomicInteger errors = new AtomicInteger();
            final AtomicInteger tokens = config.countTokens ? new AtomicInteger(0) : null;

            ExecutorService executor = Executors.newFixedThreadPool(clients);
            ArrayList<Callable<Void>> tasks = new ArrayList<>(clients);
            for (int i = 0; i < clients; ++i) {
                final int clientId = i;
                tasks.add(
                        () -> {
                            int delay = config.getDelay();
                            if (delay > 0) {
                                Thread.sleep(delay);
                            }
                            OutputStream os = config.getOutput(clientId);
                            long[] requestTime = {0L, -1L};
                            while (totalReq.getAndDecrement() > 0) {
                                SignableRequest request = new SignableRequest(serviceName, uri);
                                request.setContent(config.getRequestBody());
                                request.setHeaders(config.getRequestHeaders());
                                request.setHttpMethod(config.getRequestMethod());
                                request.setSigner(signer);
                                request.sign();
                                requestTime[0] = 0L;
                                requestTime[1] = -1L;
                                HttpResponse resp =
                                        HttpClient.sendRequest(
                                                request,
                                                insecure,
                                                config.getConnectTimeout(),
                                                os,
                                                printHeader,
                                                tokens,
                                                requestTime,
                                                config.getJsonExpression());
                                int code = resp.getStatusLine().getStatusCode();
                                if (code >= 300) {
                                    errors.getAndIncrement();
                                    continue;
                                }
                                if (requestTime[1] > 0) {
                                    firstTokens.add(requestTime[1]);
                                }

                                String token = getNextToken(resp.getFirstHeader(SM_CUSTOM_HEADER));
                                int iteration = 0;
                                while (token != null) {
                                    Thread.sleep(200);
                                    SignableRequest req = request.copy();
                                    req.addHeader(SM_CUSTOM_HEADER, "x-starting-token=" + token);
                                    req.sign();
                                    resp =
                                            HttpClient.sendRequest(
                                                    req,
                                                    insecure,
                                                    config.getConnectTimeout(),
                                                    os,
                                                    printHeader,
                                                    tokens,
                                                    requestTime,
                                                    config.getJsonExpression());
                                    code = resp.getStatusLine().getStatusCode();
                                    if (code >= 300) {
                                        break;
                                    }

                                    token = getNextToken(resp.getFirstHeader(SM_CUSTOM_HEADER));
                                    if (++iteration > 200) {
                                        System.out.println("exceed 200 pagination requests.");
                                        code = 500;
                                        break;
                                    }
                                }

                                if (code < 300) {
                                    success.add(requestTime[0]);
                                } else {
                                    errors.getAndIncrement();
                                }
                            }
                            os.close();
                            return null;
                        });
            }

            long start = System.nanoTime();
            List<Future<Void>> futures = executor.invokeAll(tasks);
            long delta = System.nanoTime() - start;

            executor.shutdown();

            for (Future<Void> future : futures) {
                try {
                    future.get();
                } catch (ExecutionException e) {
                    logger.error("", e.getCause());
                    errors.getAndIncrement();
                }
            }

            int successReq = success.size();
            int errorReq = errors.get();
            int totalRequests = successReq + errorReq;
            if (totalRequests == 0) {
                totalRequests = 1;
            }
            Collections.sort(success);
            long totalTime = success.stream().mapToLong(val -> val).sum();
            ret.setTotalTimeMills(delta / 1000000d);
            ret.setTotalRequests(totalRequests);
            ret.setFailedRequests(errorReq);
            ret.setErrorRate(100d * errorReq / totalRequests);
            ret.setConcurrentClients(clients);

            if (successReq > 0) {
                ret.setTps(successReq * 1000000000d / totalTime * clients);
                ret.setAverageLatency(totalTime / 1000000d / successReq);
                ret.setP50Latency(success.get(successReq / 2) / 1000000d);
                ret.setP90Latency(success.get(successReq * 9 / 10) / 1000000d);
                ret.setP99Latency(success.get(successReq * 99 / 100) / 1000000d);
                if (tokens != null) {
                    int totalTokens = tokens.get();
                    ret.setTotalTokens(totalTokens);
                    ret.setTokenPerRequest(totalTokens / totalRequests);
                    ret.setTokenThroughput(totalTokens * 1000000000d / totalTime * clients);
                }
            }
            if (!firstTokens.isEmpty()) {
                Collections.sort(firstTokens);
                long sum = firstTokens.stream().mapToLong(val -> val).sum();
                int size = firstTokens.size();
                ret.setTimeToFirstByte(sum / 1000000d / size);
                ret.setP50TimeToFirstByte(firstTokens.get(size / 2) / 1000000d);
                ret.setP90TimeToFirstByte(firstTokens.get(size * 9 / 10) / 1000000d);
                ret.setP99TimeToFirstByte(firstTokens.get(size * 99 / 100) / 1000000d);
            }
            AwsCurl.logger.debug("Total request time: {} ms", totalTime / 1000000d);
            ret.print(config.isJsonOutput());
        } catch (IOException | InterruptedException e) {
            System.err.println(e.getMessage());
        } catch (ParseException e) {
            System.err.println(e.getMessage());
            printHelp(jarName, options);
        }
        return ret;
    }

    private static void printHelp(String jarFileName, Options options) {
        String message;
        if (jarFileName == null) {
            message = "awscurl <URL>";
        } else {
            message = "java -jar " + jarFileName + " <URL>";
        }
        HelpFormatter formatter = new HelpFormatter();
        formatter.setLeftPadding(1);
        formatter.setWidth(120);
        formatter.printHelp(message, options);
    }

    private static String getJarName() {
        URL url = AwsCurl.class.getProtectionDomain().getCodeSource().getLocation();
        String path = url.getPath();
        if ("file".equalsIgnoreCase(url.getProtocol())) {
            File file = new File(path);
            if (path.toLowerCase().endsWith(".jar")) { // we only support jar file for now
                return file.getName();
            }
        }
        return null;
    }

    private static String inferRegion(String url, String serviceName) {
        Pattern pattern =
                Pattern.compile(
                        "http(s)?://(.+\\.)?" + serviceName + "\\.(.+)\\.amazonaws\\.com(/.*)?",
                        Pattern.CASE_INSENSITIVE);
        Matcher matcher = pattern.matcher(url);
        if (matcher.matches()) {
            return matcher.group(3);
        }

        pattern =
                Pattern.compile(
                        "http(s)?://(.+\\.)?(.+)\\." + serviceName + "\\.amazonaws\\.com(/.*)?",
                        Pattern.CASE_INSENSITIVE);
        matcher = pattern.matcher(url);
        if (matcher.matches()) {
            return matcher.group(3);
        }
        return Utils.getEnvOrSystemProperty("AWS_REGION");
    }

    static String getNextToken(Header header) {
        if (header == null) {
            return null;
        }
        String[] pair = header.getValue().split("=", 2);
        if (pair.length == 2 && "x-next-token".equalsIgnoreCase(pair[0])) {
            return pair[1];
        }
        return null;
    }

    private static final class Config {

        private String serviceName;
        private String region;
        private String profile;
        private String contentType;
        private int connectTimeout;
        private String[] data;
        private String[] dataRaw;
        private String[] dataUrlencode;
        private String[] form;
        private String[] formString;
        private String requestMethod;
        private boolean forceGet;
        private String[] headers;
        private boolean include;
        private boolean insecure;
        private boolean jsonOutput;
        private String output;
        private String uploadFile;
        private boolean verbose;
        private int nRequests;
        private int clients;
        private boolean countTokens;
        private List<byte[]> dataset;
        private String jq;
        private int delay;
        private boolean random;
        private String error;
        private AtomicInteger index;

        public Config(CommandLine cmd) throws IOException {
            dataset = new ArrayList<>();
            index = new AtomicInteger(0);
            serviceName = cmd.getOptionValue("service");
            region = cmd.getOptionValue("region");
            profile = cmd.getOptionValue("profile");
            try {
                if (cmd.hasOption("connect-timeout")) {
                    connectTimeout = Integer.parseInt(cmd.getOptionValue("connect-timeout")) * 1000;
                } else {
                    connectTimeout = 30000;
                }
            } catch (NumberFormatException e) {
                error = "Invalid connect-timeout: " + cmd.getOptionValue("connect-timeout");
            }
            String dataDirectory = cmd.getOptionValue("dataset");
            if (dataDirectory != null) {
                loadDataset(dataDirectory);
            }
            data = cmd.getOptionValues("data");
            dataRaw = cmd.getOptionValues("data-raw");
            dataUrlencode = cmd.getOptionValues("data-urlencode");
            form = cmd.getOptionValues("form");
            formString = cmd.getOptionValues("form-string");
            if (cmd.hasOption("request")) {
                requestMethod = cmd.getOptionValue("request");
            }
            forceGet = cmd.hasOption("get");
            headers = cmd.getOptionValues("header");
            include = cmd.hasOption("include");
            insecure = cmd.hasOption("insecure");
            jsonOutput = cmd.hasOption("json-output");
            output = cmd.getOptionValue("output");
            uploadFile = cmd.getOptionValue("upload-file");
            verbose = cmd.hasOption("verbose");
            if (cmd.hasOption("repeat")) {
                try {
                    nRequests = Integer.parseInt(cmd.getOptionValue("repeat"));
                } catch (NumberFormatException e) {
                    nRequests = -1;
                }
            } else {
                nRequests = 1;
            }
            if (nRequests <= 0) {
                error = "Invalid number of requests: " + cmd.getOptionValue("repeat");
            }
            if (cmd.hasOption("clients")) {
                try {
                    clients = Integer.parseInt(cmd.getOptionValue("clients"));
                } catch (NumberFormatException e) {
                    clients = -1;
                }
            } else {
                clients = 1;
            }
            if (clients <= 0) {
                error = "Invalid concurrent clients: " + cmd.getOptionValue("clients");
            }
            countTokens = cmd.hasOption("tokens");
            jq = cmd.getOptionValue("jq");
            String delayExpression = cmd.getOptionValue("delay");
            if (delayExpression != null) {
                Pattern pattern = Pattern.compile("rand\\((\\d+)( *, *(\\d+))?\\)|(\\d+)");
                Matcher m = pattern.matcher(delayExpression);
                if (m.matches()) {
                    String min = m.group(1);
                    String max = m.group(3);
                    String d = m.group(4);
                    if (d != null) {
                        delay = Integer.parseInt(d);
                    } else {
                        random = true;
                        if (max != null) {
                            delay = Integer.parseInt(max) - Integer.parseInt(min);
                        } else {
                            delay = Integer.parseInt(min);
                        }
                    }
                    if (delay < 0) {
                        delay = 0;
                    }
                }
            }
        }

        private void loadDataset(String dir) throws IOException {
            Path path = Paths.get(dir);
            if (Files.notExists(path)) {
                error = "dataset directory not found: " + dir;
                return;
            }
            if (Files.isDirectory(path)) {
                System.out.println("Loading dataset from directory: " + path);
                try (Stream<Path> stream = Files.list(path)) {
                    stream.forEach(
                            p -> {
                                try {
                                    if (Files.isRegularFile(p) && !Files.isHidden(p)) {
                                        byte[] buf = Files.readAllBytes(p);
                                        if (buf.length == 0) {
                                            System.out.println("empty dataset file: " + p);
                                        }
                                        dataset.add(buf);
                                    }
                                } catch (IOException e) {
                                    error = "Failed to read dataset file: " + p;
                                }
                            });
                }
            } else {
                System.out.println("Loading dataset from file: " + path);
                try (BufferedReader reader = Files.newBufferedReader(path)) {
                    String line;
                    while ((line = reader.readLine()) != null) {
                        dataset.add(line.getBytes(StandardCharsets.UTF_8));
                    }
                }
            }
            if (dataset.isEmpty()) {
                error = "Failed to read dataset from: " + dir;
            }
        }

        static Options getOptions() {
            Options options = new Options();
            options.addOption(
                    Option.builder("n")
                            .longOpt("service")
                            .hasArg()
                            .argName("SERVICE")
                            .desc("AWS service name")
                            .build());
            options.addOption(
                    Option.builder("r")
                            .longOpt("region")
                            .hasArg()
                            .argName("REGION")
                            .desc("AWS region name")
                            .build());
            options.addOption(
                    Option.builder("p")
                            .longOpt("profile")
                            .hasArg()
                            .argName("PROFILE")
                            .desc("AWS credentials profile name")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("connect-timeout")
                            .hasArg()
                            .argName("SECONDS")
                            .desc("Maximum time allowed for connection")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("dataset")
                            .hasArg()
                            .argName("DIRECTORY")
                            .desc("dataset directory")
                            .build());
            options.addOption(
                    Option.builder("d")
                            .longOpt("data")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data, '@' allowed")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-raw")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data (no file)")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("data-urlencode")
                            .hasArgs()
                            .argName("DATA")
                            .desc("HTTP POST data url encoded")
                            .build());
            options.addOption(
                    Option.builder("F")
                            .longOpt("form")
                            .hasArgs()
                            .argName("CONTENT")
                            .desc("Specify HTTP multipart POST data, '@' allowed")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("form-string")
                            .hasArgs()
                            .argName("STRING")
                            .desc("Specify HTTP multipart POST data (no file)")
                            .build());
            options.addOption(
                    Option.builder("G")
                            .longOpt("get")
                            .desc("Send the -d data with a HTTP GET")
                            .build());
            options.addOption(
                    Option.builder("H")
                            .longOpt("header")
                            .hasArgs()
                            .argName("LINE")
                            .desc("Pass custom header LINE to server")
                            .build());
            options.addOption(Option.builder("h").longOpt("help").desc("This help text").build());
            options.addOption(
                    Option.builder("i")
                            .longOpt("include")
                            .desc("Include protocol headers in the output")
                            .build());
            options.addOption(
                    Option.builder("k")
                            .longOpt("insecure")
                            .desc("Allow connections to SSL sites without certs")
                            .build());
            options.addOption(
                    Option.builder("o")
                            .longOpt("output")
                            .hasArg()
                            .argName("FILE")
                            .desc("Write to FILE instead of stdout")
                            .build());
            options.addOption(
                    Option.builder("X")
                            .longOpt("request")
                            .hasArg()
                            .argName("COMMAND")
                            .desc("Specify request command to use")
                            .build());
            options.addOption(
                    Option.builder("T")
                            .longOpt("upload-file")
                            .hasArg()
                            .argName("FILE")
                            .desc("Transfer FILE to destination")
                            .build());
            options.addOption(
                    Option.builder("H")
                            .longOpt("header")
                            .hasArgs()
                            .argName("LINE")
                            .desc("Pass custom header LINE to server")
                            .build());
            options.addOption(
                    Option.builder("v")
                            .longOpt("verbose")
                            .desc("Make the operation more talkative")
                            .build());
            options.addOption(
                    Option.builder("N")
                            .longOpt("repeat")
                            .hasArg()
                            .desc("Number of requests to perform")
                            .build());
            options.addOption(
                    Option.builder("c")
                            .longOpt("clients")
                            .hasArg()
                            .desc("Concurrent clients")
                            .build());
            options.addOption(
                    Option.builder("t").longOpt("tokens").desc("Output token per seconds").build());
            options.addOption(
                    Option.builder("j")
                            .longOpt("jq")
                            .hasArg()
                            .argName("EXPRESSION")
                            .desc("Json query expression for token output")
                            .build());
            options.addOption(
                    Option.builder("P")
                            .longOpt("json-output")
                            .desc("print out json format")
                            .build());
            options.addOption(
                    Option.builder()
                            .longOpt("delay")
                            .hasArg()
                            .argName("DELAY")
                            .desc(
                                    "Delay in millis for initial requests (e.g. 10 or rand(100,"
                                            + " 200))")
                            .build());
            return options;
        }

        public String getServiceName() {
            return serviceName;
        }

        public String getRegion() {
            return region;
        }

        public String getProfile() {
            return profile;
        }

        public int getConnectTimeout() {
            return connectTimeout;
        }

        public boolean isInclude() {
            return include;
        }

        public boolean isInsecure() {
            return insecure;
        }

        public boolean isJsonOutput() {
            return jsonOutput;
        }

        public OutputStream getOutput(int clientId) throws IOException {
            if (output != null) {
                return Files.newOutputStream(Paths.get(output + '.' + clientId));
            }
            if (nRequests > 1) {
                return NullOutputStream.INSTANCE;
            }
            return System.out;
        }

        public boolean isVerbose() {
            return verbose;
        }

        public int getNumberOfRequests() {
            return nRequests;
        }

        public int getClients() {
            return clients;
        }

        public String getRequestMethod() {
            if (forceGet) {
                return "GET";
            }
            return requestMethod == null ? "GET" : requestMethod;
        }

        public Map<String, String> getRequestHeaders() {
            Map<String, String> map = new ConcurrentHashMap<>();
            if (headers != null) {
                for (String header : headers) {
                    String[] pair = header.split(":", 2);
                    String key = pair[0].trim();
                    if ("content-type".equalsIgnoreCase(key)) {
                        key = "Content-Type";
                    } else if ("Content-Length".equalsIgnoreCase(key)) {
                        key = "Content-Length";
                    }
                    if (pair.length == 2) {
                        map.put(key, pair[1].trim());
                    }
                }
            }
            if (contentType != null) {
                map.putIfAbsent("Content-Type", contentType);
            }

            return map;
        }

        public String getUrl(String url) throws IOException {
            if (!forceGet) {
                return url;
            }

            if (data != null || dataRaw != null || dataUrlencode != null) {
                contentType = null;
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                addUrlEncodedData(bos, data, 1);
                addUrlEncodedData(bos, dataRaw, 2);
                addUrlEncodedData(bos, dataUrlencode, 3);
                bos.close();
                String queryString = bos.toString(StandardCharsets.UTF_8);
                int pos = url.indexOf('?');
                if (pos >= 0) {
                    if (pos == url.length() - 1) {
                        return url + queryString;
                    }
                    return url + '&' + queryString;
                }
                return url + '?' + queryString;
            }
            return url;
        }

        @SuppressWarnings("PMD.ReturnEmptyCollectionRatherThanNull")
        public byte[] getRequestBody() throws IOException {
            if (forceGet) {
                return null;
            }

            /*
             * Priority:
             *  1. --dataset
             *  2. --form, --form-string
             *  3. --data, --data-raw, --data-urlencode
             *  4. --upload-file
             */
            if (!dataset.isEmpty()) {
                int i = index.incrementAndGet() % dataset.size();
                return dataset.get(i);
            }

            if (form != null || formString != null) {
                requestMethod = requestMethod == null ? "POST" : requestMethod;
                MultipartEntityBuilder mb = MultipartEntityBuilder.create();
                addFormPart(mb, form, true);
                addFormPart(mb, formString, false);
                HttpEntity entity = mb.build();
                contentType = entity.getContentType().getValue();

                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                entity.writeTo(bos);
                return bos.toByteArray();
            }

            if (data != null || dataRaw != null || dataUrlencode != null) {
                requestMethod = requestMethod == null ? "POST" : requestMethod;
                contentType = ContentType.APPLICATION_FORM_URLENCODED.toString();
                ByteArrayOutputStream bos = new ByteArrayOutputStream();
                addUrlEncodedData(bos, data, 1);
                addUrlEncodedData(bos, dataRaw, 2);
                addUrlEncodedData(bos, dataUrlencode, 3);
                bos.close();
                return bos.toByteArray();
            }

            if (uploadFile != null) {
                requestMethod = requestMethod == null ? "PUT" : requestMethod;
                if (headers != null) {
                    for (String header : headers) {
                        String[] pair = header.split(":", 2);
                        String key = pair[0].trim();
                        if ("content-type".equalsIgnoreCase(key)) {
                            contentType = pair[1].trim();
                        }
                    }
                }
                if (contentType == null) {
                    contentType = getMimeType(uploadFile).toString();
                }
                return readFile(uploadFile);
            }

            return null;
        }

        public String[] getJsonExpression() {
            if (jq == null) {
                return null; // NOPMD
            }
            return jq.split("/");
        }

        public int getDelay() {
            if (random) {
                RandomUtils.nextInt(delay);
            }
            return delay;
        }

        public String getError() {
            return error;
        }

        private byte[] readFile(String fileName) throws IOException {
            Path path = Paths.get(fileName);
            if (!Files.isRegularFile(path)) {
                throw new FileNotFoundException("File not found: " + fileName);
            }
            try (InputStream is = Files.newInputStream(path)) {
                return Utils.toByteArray(is);
            }
        }

        private void addUrlEncodedData(ByteArrayOutputStream bos, String[] data, int encodeType)
                throws IOException {
            if (data == null) {
                return;
            }
            for (String content : data) {
                switch (encodeType) {
                    case 1: // data
                        if (content.startsWith("@")) {
                            String dataFile = content.substring(1);
                            Path path = Paths.get(dataFile);
                            if (!Files.isRegularFile(path)) {
                                throw new IOException("Invalid input file: " + dataFile);
                            }
                            Files.copy(path, bos);
                        } else {
                            bos.write(content.getBytes(StandardCharsets.UTF_8));
                        }
                        break;
                    case 2: // data-raw
                        bos.write(content.getBytes(StandardCharsets.UTF_8));
                        break;
                    case 3:
                    default:
                        writeUrlEncodedData(bos, content);
                        break;
                }
            }
        }

        private void writeUrlEncodedData(ByteArrayOutputStream bos, String content)
                throws IOException {
            String[] parameters = content.split("&");
            for (String parameter : parameters) {
                if (bos.size() > 0) {
                    bos.write('&');
                }

                String[] pair = parameter.split("=", 2);
                if (pair.length == 1) {
                    bos.write(readContentUrlEncoded(pair[0]));
                } else if (pair[0].isEmpty()) {
                    bos.write(readContentUrlEncoded(pair[1]));
                } else {
                    bos.write(pair[0].getBytes(StandardCharsets.UTF_8));
                    bos.write('=');
                    bos.write(readContentUrlEncoded(pair[1]));
                }
            }
        }

        private byte[] readContentUrlEncoded(String content) throws IOException {
            if (content.startsWith("@")) {
                Path file = Paths.get(content.substring(1));
                String value = Files.readString(file);
                return URLEncoder.encode(value, StandardCharsets.UTF_8)
                        .getBytes(StandardCharsets.UTF_8);
            }
            return URLEncoder.encode(content, StandardCharsets.UTF_8)
                    .getBytes(StandardCharsets.UTF_8);
        }

        private void addFormPart(MultipartEntityBuilder mb, String[] forms, boolean allowFile) {
            if (forms == null) {
                return;
            }

            for (String parameter : forms) {
                String[] tokens = parameter.split(";");
                String key = null;
                String value = null;
                String type = null;
                String fileName = null;
                for (String token : tokens) {
                    String[] pair = token.split("=", 2);
                    if ("type".equalsIgnoreCase(pair[0])) {
                        if (pair.length > 1) {
                            type = pair[1];
                        }
                    } else if ("filename".equalsIgnoreCase(pair[0])) {
                        if (pair.length > 1) {
                            fileName = pair[1];
                        }
                    } else {
                        key = pair[0];
                        value = pair.length > 1 ? pair[1] : "";
                    }
                }
                if (key == null || value == null) {
                    AwsCurl.logger.warn("Ignore invalid form data: {}", parameter);
                    continue;
                }

                if (allowFile && value.startsWith("@")) {
                    File file = new File(value.substring(1));
                    if (StringUtils.isEmpty(fileName)) {
                        fileName = file.getName();
                    }
                    ContentType ct;
                    if (StringUtils.isEmpty(type)) {
                        ct = getMimeType(fileName);
                    } else {
                        ct = ContentType.create(type);
                    }
                    mb.addBinaryBody(key, file, ct, fileName);
                } else {
                    ContentType ct;
                    if (StringUtils.isEmpty(type)) {
                        ct = ContentType.TEXT_PLAIN;
                    } else {
                        ct = ContentType.create(type);
                    }
                    mb.addTextBody(key, value, ct);
                }
            }
        }

        private ContentType getMimeType(String fileName) {
            String ext = FilenameUtils.getFileType(fileName).toLowerCase(Locale.ROOT);
            switch (ext.toLowerCase(Locale.ENGLISH)) {
                case "txt":
                case "text":
                    return ContentType.TEXT_PLAIN;
                case "html":
                    return ContentType.TEXT_HTML;
                case "xhtml":
                    return ContentType.APPLICATION_XHTML_XML;
                case "xml":
                    return ContentType.APPLICATION_XML;
                case "json":
                    return ContentType.APPLICATION_JSON;
                case "jpg":
                case "jpeg":
                    return ContentType.IMAGE_JPEG;
                case "png":
                    return ContentType.IMAGE_PNG;
                case "gif":
                    return ContentType.IMAGE_GIF;
                case "bmp":
                    return ContentType.IMAGE_BMP;
                case "svg":
                    return ContentType.IMAGE_SVG;
                case "tiff":
                    return ContentType.IMAGE_TIFF;
                case "webp":
                    return ContentType.IMAGE_WEBP;
                default:
                    return ContentType.APPLICATION_OCTET_STREAM;
            }
        }
    }
}
