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
package ai.djl.serving.http;

import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.serving.models.Endpoint;
import ai.djl.serving.models.ModelManager;
import ai.djl.serving.wlm.ModelInfo;
import ai.djl.serving.wlm.WorkerPoolConfig;
import ai.djl.serving.workflow.Workflow;
import ai.djl.util.Utils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.net.URLEncoder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.InvalidPathException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SortedMap;
import java.util.SortedSet;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BooleanSupplier;
import java.util.function.LongSupplier;
import java.util.function.Supplier;
import java.util.regex.Pattern;

/**
 * Collects the metrics that inference engines record inside their Python worker processes.
 *
 * <p>Each worker that serves its metrics registers the path of a Unix domain socket as a model
 * property named {@code engine_metrics_socket.<label>}. A scrape reads the registrations, fetches
 * every socket concurrently and merges the answers into one Prometheus text exposition.
 *
 * <p>Concurrent scrapes with the same arguments share one fetch, and a finished fetch is reused for
 * one second, so scrapers that send the same request do not add requests for the workers. The raw
 * answers are what is shared. Each caller merges them itself. Only the most recent fetch is kept: a
 * scrape with other arguments starts a new fetch, and scrapers that alternate between different
 * arguments each cause one. At most {@link #MAX_UNFINISHED} fetches may be unfinished at once. A
 * scrape that would start another one is refused.
 */
final class EngineMetrics {

    /** The prefix of the model property that holds the socket path of one worker. */
    static final String KEY_PREFIX = "engine_metrics_socket.";

    static final String SOCKET_SUFFIX = ".metrics";
    static final int MAX_BODY_BYTES = 4 * 1024 * 1024;
    static final long TIMEOUT_MS = 2000;
    static final long REUSE_NANOS = TimeUnit.SECONDS.toNanos(1);
    static final long LOG_INTERVAL_NANOS = TimeUnit.MINUTES.toNanos(5);

    /**
     * The age of a fetch that has not finished and will not: the client gives up on a source after
     * {@link #TIMEOUT_MS}, so a fetch that is far older has lost its completion.
     */
    static final long STUCK_NANOS = TimeUnit.MILLISECONDS.toNanos(4 * TIMEOUT_MS);

    /**
     * The most fetches that may be unfinished at once. Each one holds a connection in the queue of
     * every worker, and a worker renders the queued requests one at a time, also those that DJL
     * Serving has given up on. This bounds how long the next scrape waits behind them and how much
     * text is held. A scrape that would start one more fetch is refused.
     */
    static final int MAX_UNFINISHED = 32;

    private static final Logger logger = LoggerFactory.getLogger(EngineMetrics.class);

    private static final Pattern LABEL = Pattern.compile("[A-Za-z0-9._-]{1,64}");

    private static final Pattern METRIC_NAME_PATTERN =
            Pattern.compile("^[a-zA-Z_:][a-zA-Z0-9_:]*$");

    private static final EngineMetrics INSTANCE =
            new EngineMetrics(
                    EngineMetrics::sources,
                    () -> "true".equals(Utils.getenv("SAGEMAKER_MULTI_MODEL")),
                    EngineMetrics::fetch,
                    System::nanoTime,
                    logger);

    private final Supplier<SortedMap<String, String>> sourceSupplier;
    private final BooleanSupplier multiModel;
    private final Fetcher fetcher;
    private final LongSupplier clock;
    private final Logger log;

    private final Object lock = new Object();
    private Flight current;

    /** The fetches that have not finished, guarded by {@link #lock}. */
    private final List<Flight> unfinished = new ArrayList<>();

    /** The refused scrapes since the last message, and when that was logged, under the lock. */
    private int refused;

    private boolean refusalLogged;
    private long refusalLoggedAt;

    private final Map<String, Health> health = new ConcurrentHashMap<>();

    EngineMetrics(
            Supplier<SortedMap<String, String>> sourceSupplier,
            BooleanSupplier multiModel,
            Fetcher fetcher,
            LongSupplier clock,
            Logger log) {
        this.sourceSupplier = sourceSupplier;
        this.multiModel = multiModel;
        this.fetcher = fetcher;
        this.clock = clock;
        this.log = log;
    }

    /**
     * Returns {@code true} if engine metrics are switched on.
     *
     * @return {@code true} if {@code SERVING_ENGINE_METRICS} is true
     */
    static boolean isEnabled() {
        return Boolean.parseBoolean(Utils.getEnvOrSystemProperty("SERVING_ENGINE_METRICS"));
    }

    /**
     * Returns the registered sockets of all loaded models.
     *
     * @return the socket path of each registered worker, by worker label
     */
    static SortedMap<String, String> sources() {
        return sources(ModelManager.getInstance().getEndpoints().values());
    }

    /**
     * Returns the sockets that the workers of the given endpoints registered.
     *
     * <p>A registration is accepted only if the path is inside the temporary directory and ends in
     * {@code .metrics}, and the label is short and made of letters, digits, dots, underscores and
     * hyphens. Model properties can come from a configuration file, and this keeps them from
     * pointing the server at an arbitrary socket or writing odd text into logs and queries.
     *
     * @param endpoints the endpoints to read
     * @return the socket path of each registered worker, by worker label
     */
    static SortedMap<String, String> sources(Collection<Endpoint> endpoints) {
        SortedMap<String, String> found = new TreeMap<>();
        Path tmp = Paths.get(System.getProperty("java.io.tmpdir")).toAbsolutePath().normalize();
        for (Endpoint endpoint : endpoints) {
            List<Workflow> workflows;
            synchronized (endpoint) {
                // add() and remove() hold the endpoint lock and the list is a plain ArrayList
                workflows = new ArrayList<>(endpoint.getWorkflows());
            }
            for (Workflow workflow : workflows) {
                if (workflow == null) {
                    continue;
                }
                for (WorkerPoolConfig<Input, Output> wpc : workflow.getWpcs()) {
                    if (wpc instanceof ModelInfo) {
                        ModelInfo<Input, Output> info = (ModelInfo<Input, Output>) wpc;
                        for (ZooModel<Input, Output> model : info.getModels().values()) {
                            collect(found, tmp, model.getProperties());
                        }
                    }
                }
            }
        }
        return found;
    }

    /**
     * Fetches the engine metrics of all registered workers.
     *
     * @param names the metric names to keep, empty to keep all
     * @return the merged text and how many workers were registered and answered
     */
    static CompletableFuture<Result> scrape(List<String> names) {
        return INSTANCE.scrapeSources(names);
    }

    /**
     * Merges Prometheus text expositions into one.
     *
     * <p>Zero or one body is returned unchanged. Otherwise lines are grouped by metric family in
     * the order the families are first seen. A family keeps the {@code HELP} and {@code TYPE} lines
     * of the first body that has them, followed by the sample lines of every body in body order.
     * Samples that precede the first header of their body go to a group without a header. Other
     * comment lines and blank lines are dropped.
     *
     * @param bodies the expositions, in the order their samples should appear
     * @return the merged exposition
     */
    static String merge(List<String> bodies) {
        if (bodies.isEmpty()) {
            return "";
        } else if (bodies.size() == 1) {
            return bodies.get(0);
        }
        LinkedHashMap<String, Family> families = new LinkedHashMap<>();
        for (String body : bodies) {
            Family family = null;
            int start = 0;
            int length = body.length();
            while (start < length) {
                int end = body.indexOf('\n', start);
                if (end < 0) {
                    end = length;
                }
                String line = body.substring(start, end);
                start = end + 1;
                if (line.endsWith("\r")) {
                    line = line.substring(0, line.length() - 1);
                }
                if (line.isBlank()) {
                    continue;
                }
                boolean help = line.startsWith("# HELP ");
                if (help || line.startsWith("# TYPE ")) {
                    int nameEnd = line.indexOf(' ', 7);
                    String name = nameEnd < 0 ? line.substring(7) : line.substring(7, nameEnd);
                    family = families.computeIfAbsent(name, k -> new Family());
                    family.addHeader(line, help);
                } else if (line.charAt(0) != '#') {
                    if (family == null) {
                        family = families.computeIfAbsent("", k -> new Family());
                    }
                    family.samples.add(line);
                }
            }
        }
        StringBuilder sb = new StringBuilder(8192);
        for (Family family : families.values()) {
            for (String line : family.headers) {
                sb.append(line).append('\n');
            }
            for (String line : family.samples) {
                sb.append(line).append('\n');
            }
        }
        return sb.toString();
    }

    /**
     * Returns the request target for one worker.
     *
     * @param label the worker label
     * @param workerLabel true to ask the worker to add its label to every sample
     * @param names the metric names to keep, empty to keep all
     * @return the request target
     */
    static String uri(String label, boolean workerLabel, Collection<String> names) {
        StringBuilder sb = new StringBuilder(64).append("/metrics");
        char separator = '?';
        if (workerLabel) {
            sb.append(separator).append("worker=").append(encode(label));
            separator = '&';
        }
        for (String name : names) {
            sb.append(separator).append("name%5B%5D=").append(encode(name));
            separator = '&';
        }
        return sb.toString();
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
    }

    private static void collect(
            SortedMap<String, String> found, Path tmp, Map<String, String> properties) {
        if (properties == null) {
            return;
        }
        for (Map.Entry<String, String> entry : properties.entrySet()) {
            String key = entry.getKey();
            if (key.startsWith(KEY_PREFIX)) {
                String label = key.substring(KEY_PREFIX.length());
                String socket = trusted(tmp, entry.getValue());
                if (socket != null && LABEL.matcher(label).matches()) {
                    found.putIfAbsent(label, socket);
                }
            }
        }
    }

    /**
     * Returns the normalized socket path if it is absolute, inside the temporary directory and ends
     * in {@code .metrics}, otherwise {@code null}. The normalized form is what gets connected to,
     * so the check applies to the path that is used. A relative path is refused: it would be
     * resolved against the working directory of the server, which is not where the worker made the
     * socket.
     */
    private static String trusted(Path tmp, String value) {
        if (value == null) {
            return null;
        }
        try {
            Path given = Paths.get(value);
            if (!given.isAbsolute()) {
                return null;
            }
            Path path = given.normalize();
            Path name = path.getFileName();
            if (path.startsWith(tmp) && name != null && name.toString().endsWith(SOCKET_SUFFIX)) {
                return path.toString();
            }
        } catch (InvalidPathException e) {
            return null;
        }
        return null;
    }

    /**
     * Fetches one source. Symbolic links are resolved first, and the request goes to the resolved
     * path only if it is still inside the temporary directory, so a link placed there cannot point
     * the server at another socket.
     */
    static CompletableFuture<String> fetch(String socketPath, String uri) {
        Path path = Paths.get(socketPath);
        if (!Files.exists(path)) {
            return CompletableFuture.failedFuture(new NoSuchFileException(socketPath));
        }
        Path real;
        try {
            real = path.toRealPath();
            Path tmp = Paths.get(System.getProperty("java.io.tmpdir")).toRealPath();
            if (!real.startsWith(tmp)) {
                return CompletableFuture.failedFuture(
                        new IOException(
                                "Engine metrics socket resolves outside the temporary directory: "
                                        + socketPath));
            }
        } catch (IOException e) {
            return CompletableFuture.failedFuture(e);
        }
        return EngineMetricsClient.get(real.toString(), uri, MAX_BODY_BYTES, TIMEOUT_MS);
    }

    CompletableFuture<Result> scrapeSources(List<String> names) {
        Key key;
        try {
            SortedMap<String, String> registered = sourceSupplier.get();
            boolean workerLabel = registered.size() > 1 || multiModel.getAsBoolean();
            key = new Key(registered, workerLabel, names);
        } catch (RuntimeException e) {
            return CompletableFuture.failedFuture(e);
        }
        if (key.names.isEmpty() && names.stream().anyMatch(Objects::nonNull)) {
            // Every requested name was invalid: nothing can match, so do not fetch at all.
            return CompletableFuture.completedFuture(new Result("", 0, 0));
        }
        Flight flight;
        boolean start = false;
        int reportRefused = 0;
        synchronized (lock) {
            long now = clock.getAsLong();
            flight = current;
            if (flight == null || !flight.key.equals(key) || flight.expired(now)) {
                unfinished.removeIf(f -> f.done.isDone() || f.lost(now));
                if (unfinished.size() < MAX_UNFINISHED) {
                    flight = new Flight(key, now);
                    unfinished.add(flight);
                    current = flight;
                    start = true;
                } else {
                    flight = null;
                    reportRefused = countRefusal(now);
                }
            }
        }
        if (flight == null) {
            if (reportRefused > 0) {
                log.warn(
                        "Engine metrics: {} fetches are unfinished, refusing scrapes that need"
                                + " another one ({} refused since the last message)",
                        MAX_UNFINISHED,
                        reportRefused);
            }
            return CompletableFuture.failedFuture(
                    new ServiceUnavailableException(
                            "Engine metrics are busy: "
                                    + MAX_UNFINISHED
                                    + " fetches are unfinished, try again later."));
        }
        if (start) {
            try {
                fetchAll(flight);
            } catch (Throwable t) {
                // A flight that never completes would keep every caller with the same key waiting.
                flight.completedAt.set(clock.getAsLong());
                flight.done.completeExceptionally(t);
                throw t;
            }
        }
        return flight.done.thenApply(EngineMetrics::toResult);
    }

    /**
     * Counts a refused scrape. Returns the number of refused scrapes to report if a message is due,
     * at most one per {@link #LOG_INTERVAL_NANOS}, otherwise 0. Call it with the lock held.
     */
    private int countRefusal(long now) {
        ++refused;
        if (refusalLogged && now - refusalLoggedAt < LOG_INTERVAL_NANOS) {
            return 0;
        }
        int count = refused;
        refused = 0;
        refusalLogged = true;
        refusalLoggedAt = now;
        return count;
    }

    private void fetchAll(Flight flight) {
        Key key = flight.key;
        List<CompletableFuture<Fetched>> parts = new ArrayList<>();
        for (Map.Entry<String, String> source : key.sources.entrySet()) {
            String label = source.getKey();
            CompletableFuture<String> pending;
            try {
                pending = fetcher.get(source.getValue(), uri(label, key.workerLabel, key.names));
            } catch (RuntimeException e) {
                pending = CompletableFuture.failedFuture(e);
            }
            parts.add(pending.handle((body, error) -> new Fetched(label, body, error)));
        }
        CompletableFuture.allOf(parts.toArray(new CompletableFuture<?>[0]))
                .whenComplete(
                        (v, error) -> {
                            List<Fetched> fetched = new ArrayList<>();
                            for (CompletableFuture<Fetched> part : parts) {
                                fetched.add(part.join());
                            }
                            try {
                                report(key.sources.keySet(), fetched);
                            } finally {
                                flight.completedAt.set(clock.getAsLong());
                                flight.done.complete(fetched);
                            }
                        });
    }

    private static Result toResult(List<Fetched> fetched) {
        List<String> bodies = new ArrayList<>();
        for (Fetched part : fetched) {
            if (part.error == null) {
                bodies.add(part.body == null ? "" : part.body);
            }
        }
        return new Result(merge(bodies), fetched.size(), bodies.size());
    }

    /**
     * Logs the failure of a source at most once per {@link #LOG_INTERVAL_NANOS}, and the recovery
     * of a failure that was logged. A source that keeps failing is reminded of once per interval. A
     * source that goes up and down is reported once per interval as well, not once per cycle.
     * Nothing is logged for a scrape that behaves like the previous one.
     */
    private void report(Collection<String> labels, List<Fetched> fetched) {
        long now = clock.getAsLong();
        synchronized (health) {
            health.keySet().retainAll(labels);
            for (Fetched part : fetched) {
                Health state = health.computeIfAbsent(part.label, k -> new Health());
                if (part.error == null) {
                    if (state.failing && state.reported) {
                        log.info("Engine metrics source {} recovered", part.label);
                    }
                    state.failing = false;
                    state.reported = false;
                } else {
                    if (!state.logged || now - state.loggedAt >= LOG_INTERVAL_NANOS) {
                        logFailure(state, part);
                        state.reported = true;
                        state.logged = true;
                        state.loggedAt = now;
                        state.suppressed = 0;
                    } else {
                        ++state.suppressed;
                    }
                    state.failing = true;
                }
            }
        }
    }

    private void logFailure(Health state, Fetched part) {
        String reason = EngineMetricsClient.describe(part.error);
        if (state.failing) {
            log.warn(
                    "Engine metrics source {} is still failing: {} ({} failed fetches since the"
                            + " last message)",
                    part.label,
                    reason,
                    state.suppressed + 1);
        } else if (state.suppressed > 0) {
            log.warn(
                    "Engine metrics source {} failed: {} ({} failed fetches since the last"
                            + " message)",
                    part.label,
                    reason,
                    state.suppressed + 1);
        } else {
            log.warn("Engine metrics source {} failed: {}", part.label, reason);
        }
    }

    /** Fetches one source. */
    @FunctionalInterface
    interface Fetcher {

        /**
         * Returns the exposition of one worker.
         *
         * @param socketPath the socket of the worker
         * @param uri the request target
         * @return a future of the response body
         */
        CompletableFuture<String> get(String socketPath, String uri);
    }

    /** The outcome of a scrape. */
    static final class Result {

        /** The merged exposition of the workers that answered. */
        final String body;

        /** The number of registered workers. */
        final int registered;

        /** The number of workers that answered. */
        final int ok;

        Result(String body, int registered, int ok) {
            this.body = body;
            this.registered = registered;
            this.ok = ok;
        }
    }

    /** What makes two scrapes interchangeable. */
    private static final class Key {

        final SortedMap<String, String> sources;
        final boolean workerLabel;
        final SortedSet<String> names;

        Key(SortedMap<String, String> sources, boolean workerLabel, Collection<String> names) {
            this.sources = new TreeMap<>(sources);
            this.workerLabel = workerLabel;
            this.names = new TreeSet<>();
            for (String name : names) {
                if (name != null && METRIC_NAME_PATTERN.matcher(name).matches()) {
                    this.names.add(name);
                }
            }
        }

        /** {@inheritDoc} */
        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            Key other = (Key) o;
            return workerLabel == other.workerLabel
                    && sources.equals(other.sources)
                    && names.equals(other.names);
        }

        /** {@inheritDoc} */
        @Override
        public int hashCode() {
            return Objects.hash(sources, workerLabel, names);
        }
    }

    /** One fetch of all sources, shared by the scrapes that have the same key. */
    private static final class Flight {

        final Key key;
        final long startedAt;
        final CompletableFuture<List<Fetched>> done = new CompletableFuture<>();

        /** When {@link #done} completed, set before it completes. */
        final AtomicLong completedAt = new AtomicLong();

        Flight(Key key, long startedAt) {
            this.key = key;
            this.startedAt = startedAt;
        }

        /**
         * Returns {@code true} if a caller must not share this flight: it finished longer ago than
         * the reuse window, or it is lost. The window starts when the fetch finishes, so a slow
         * fetch is still reused for the whole window.
         */
        boolean expired(long now) {
            if (done.isDone()) {
                return now - completedAt.get() >= REUSE_NANOS;
            }
            return lost(now);
        }

        /**
         * Returns {@code true} if the flight is not finished long after the client's own deadlines,
         * which means that its completion was lost. It does not count as unfinished any more.
         */
        boolean lost(long now) {
            return !done.isDone() && now - startedAt >= STUCK_NANOS;
        }
    }

    /** The raw answer of one source: a body or the reason there is none. */
    private static final class Fetched {

        final String label;
        final String body;
        final Throwable error;

        Fetched(String label, String body, Throwable error) {
            this.label = label;
            this.body = body;
            this.error = error;
        }
    }

    /** The log state of one source. */
    private static final class Health {

        /** The previous scrape of the source failed. */
        boolean failing;

        /** The current failure was logged, so its recovery is logged too. */
        boolean reported;

        /** A failure of the source was logged at {@link #loggedAt}. */
        boolean logged;

        long loggedAt;

        /** The failures since the last message. */
        int suppressed;
    }

    /** The lines of one metric family. */
    private static final class Family {

        final List<String> headers = new ArrayList<>(2);
        final List<String> samples = new ArrayList<>();
        private boolean hasHelp;
        private boolean hasType;

        void addHeader(String line, boolean help) {
            if (help && !hasHelp) {
                hasHelp = true;
                headers.add(line);
            } else if (!help && !hasType) {
                hasType = true;
                headers.add(line);
            }
        }
    }
}
