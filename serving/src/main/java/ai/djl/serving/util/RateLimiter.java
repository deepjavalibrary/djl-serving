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
package ai.djl.serving.util;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicLong;

/** A rate limiter distributes permits at a configurable rate. */
public class RateLimiter {

    private long threshold;
    private long timeWindow;
    private AtomicLong tokens;
    private long lastRefillTimestamp;

    /**
     * Constructs a {@code RateLimiter} with the specified threshold.
     *
     * @param threshold the maximum number of tokens that can be accumulated
     * @param timeWindow the limit time window
     */
    public RateLimiter(long threshold, Duration timeWindow) {
        this.threshold = threshold;
        this.timeWindow = timeWindow.toMillis();
        tokens = new AtomicLong(threshold);
        lastRefillTimestamp = System.currentTimeMillis();
    }

    /**
     * Obtains a {@code RateLimiter} from a text string.
     *
     * @param limit the string representation of the {@code RateLimiter}
     * @return a instance of {@code RateLimiter}
     */
    public static RateLimiter parse(String limit) {
        String[] pair = limit.split("/", 2);
        long threshold = Long.parseLong(pair[0]);
        Duration duration;
        if (pair.length == 2) {
            duration = Duration.parse(pair[1]);
        } else {
            duration = Duration.ofMinutes(1);
        }
        return new RateLimiter(threshold, duration);
    }

    /**
     * Check if rate limit is hit.
     *
     * @return true if rate limit is hit
     */
    public boolean exceed() {
        long now = System.currentTimeMillis();
        long tokensToAdd = ((now - lastRefillTimestamp) / timeWindow) * threshold;

        if (tokensToAdd > 0) {
            tokens.set(Math.min(threshold, tokens.addAndGet(tokensToAdd)));
            lastRefillTimestamp = now;
        }

        if (tokens.get() > 0) {
            tokens.decrementAndGet();
            return false;
        }
        return true;
    }
}
