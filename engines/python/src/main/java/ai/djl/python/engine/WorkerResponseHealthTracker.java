/*
 * Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.python.engine;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.function.LongSupplier;

/**
 * Tracks the rolling count of non-200 (error) response codes returned by a single Python worker
 * over a sliding time window.
 *
 * <p>This exists to detect a worker that is <em>alive</em> (its process and socket are healthy) but
 * that returns an application-level error code (for example HTTP 424 on a {@code BrokenProcessPool})
 * on every request. Such a worker is never caught by {@link PyProcess#stopPythonProcess(boolean)},
 * which only increments the {@code failed} counter on process/socket death.
 *
 * <p>The tracker is intentionally free of any dependency on {@code Model}, sockets or timers so it
 * can be unit tested deterministically with an injected {@link LongSupplier} clock. All decisions
 * about what to <em>do</em> when the breaker trips (increment the {@code failed} counter, restart
 * the worker, emit a metric) live in {@link PyProcess}.
 *
 * <p>The feature is disabled by default and is fully backward compatible: when both {@code
 * threshold} is not positive and {@code metricEnabled} is {@code false}, {@link #enabled()} returns
 * {@code false} and the caller performs no additional work.
 */
final class WorkerResponseHealthTracker {

    /** Response codes at or above this value are treated as worker errors (matches 424 / 5xx). */
    static final int ERROR_CODE_THRESHOLD = 300;

    private final int threshold;
    private final long windowMillis;
    private final boolean metricEnabled;
    private final LongSupplier clock;
    private final Deque<Long> errorTimestamps;

    /**
     * Creates a tracker using the wall clock.
     *
     * @param threshold the number of error responses within the window that trips the breaker; a
     *     value {@code <= 0} disables the breaker action
     * @param windowMillis the length of the sliding window in milliseconds
     * @param metricEnabled whether a per-worker error metric should be emitted even when the breaker
     *     action is disabled
     */
    WorkerResponseHealthTracker(int threshold, long windowMillis, boolean metricEnabled) {
        this(threshold, windowMillis, metricEnabled, System::currentTimeMillis);
    }

    /**
     * Creates a tracker with an injectable clock (used by tests for deterministic window behavior).
     *
     * @param threshold the number of error responses within the window that trips the breaker; a
     *     value {@code <= 0} disables the breaker action
     * @param windowMillis the length of the sliding window in milliseconds
     * @param metricEnabled whether a per-worker error metric should be emitted even when the breaker
     *     action is disabled
     * @param clock supplies the current time in milliseconds
     */
    WorkerResponseHealthTracker(
            int threshold, long windowMillis, boolean metricEnabled, LongSupplier clock) {
        this.threshold = threshold;
        this.windowMillis = windowMillis <= 0 ? 1 : windowMillis;
        this.metricEnabled = metricEnabled;
        this.clock = clock;
        this.errorTimestamps = new ArrayDeque<>();
    }

    /**
     * Returns whether the tracker needs to do any work at all. When this is {@code false} the caller
     * behaves exactly as it did before this feature existed.
     *
     * @return {@code true} if either the breaker or the metric is enabled
     */
    boolean enabled() {
        return threshold > 0 || metricEnabled;
    }

    /**
     * Returns whether a per-worker error metric should be emitted.
     *
     * @return {@code true} if metric emission is enabled
     */
    boolean metricEnabled() {
        return metricEnabled;
    }

    /**
     * Returns whether a response code counts as a worker error.
     *
     * @param code the response code
     * @return {@code true} if the code is an error (424 / 5xx / 3xx+)
     */
    static boolean isErrorCode(int code) {
        return code >= ERROR_CODE_THRESHOLD;
    }

    /**
     * Records a single response code and reports whether the breaker should trip now.
     *
     * <p>Old entries outside the sliding window are pruned first. A success (non-error) code prunes
     * the window but is not itself recorded, so a worker that recovers gradually walks its error
     * count back down. The breaker only trips when the breaker action is enabled ({@code threshold >
     * 0}) and the number of errors within the window reaches the threshold.
     *
     * @param code the response code returned by the worker
     * @return {@code true} if the breaker should trip as a result of this record
     */
    synchronized boolean record(int code) {
        long now = clock.getAsLong();
        pruneOlderThan(now - windowMillis);
        if (isErrorCode(code)) {
            errorTimestamps.addLast(now);
        }
        return threshold > 0 && errorTimestamps.size() >= threshold;
    }

    /**
     * Returns the number of error responses currently within the sliding window.
     *
     * @return the current windowed error count
     */
    synchronized int windowErrorCount() {
        pruneOlderThan(clock.getAsLong() - windowMillis);
        return errorTimestamps.size();
    }

    /** Clears the recorded errors, for example after the breaker trips. */
    synchronized void reset() {
        errorTimestamps.clear();
    }

    private void pruneOlderThan(long cutoff) {
        while (!errorTimestamps.isEmpty() && errorTimestamps.peekFirst() < cutoff) {
            errorTimestamps.pollFirst();
        }
    }
}
