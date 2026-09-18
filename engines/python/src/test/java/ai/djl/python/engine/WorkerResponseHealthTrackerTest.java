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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.concurrent.atomic.AtomicLong;

/** Deterministic unit tests for {@link WorkerResponseHealthTracker}. */
public class WorkerResponseHealthTrackerTest {

    @Test
    public void testPersistentErrorsCrossThreshold() {
        // threshold 3 within a 60s window; a live worker returns 424 on every request.
        AtomicLong clock = new AtomicLong(0);
        WorkerResponseHealthTracker tracker =
                new WorkerResponseHealthTracker(3, 60_000L, false, clock::get);

        Assert.assertTrue(tracker.enabled());
        Assert.assertFalse(tracker.record(424), "1st 424 must not trip");
        Assert.assertFalse(tracker.record(424), "2nd 424 must not trip");
        Assert.assertTrue(tracker.record(424), "3rd 424 within window must trip the breaker");
        Assert.assertEquals(tracker.windowErrorCount(), 3);
    }

    @Test
    public void testBelowThresholdNoTrip() {
        AtomicLong clock = new AtomicLong(0);
        WorkerResponseHealthTracker tracker =
                new WorkerResponseHealthTracker(3, 60_000L, false, clock::get);

        Assert.assertFalse(tracker.record(424));
        Assert.assertFalse(tracker.record(500));
        // A success prunes the window but is not itself an error; still below threshold.
        Assert.assertFalse(tracker.record(200));
        Assert.assertEquals(tracker.windowErrorCount(), 2);
    }

    @Test
    public void testDisabledFeatureNeverTrips() {
        // threshold <= 0 and metric disabled == fully disabled == current behavior unchanged.
        WorkerResponseHealthTracker tracker = new WorkerResponseHealthTracker(0, 60_000L, false);
        Assert.assertFalse(tracker.enabled());
        Assert.assertFalse(tracker.metricEnabled());
        for (int i = 0; i < 100; ++i) {
            Assert.assertFalse(tracker.record(424), "disabled breaker must never trip");
        }
    }

    @Test
    public void testMetricOnlyEnabledDoesNotTrip() {
        // metric enabled but threshold disabled: tracker is "enabled" for metric emission but the
        // breaker never trips regardless of how many errors arrive.
        WorkerResponseHealthTracker tracker = new WorkerResponseHealthTracker(0, 60_000L, true);
        Assert.assertTrue(tracker.enabled());
        Assert.assertTrue(tracker.metricEnabled());
        Assert.assertFalse(tracker.record(424));
        Assert.assertFalse(tracker.record(424));
        Assert.assertFalse(tracker.record(424));
    }

    @Test
    public void testErrorsOutsideWindowArePruned() {
        AtomicLong clock = new AtomicLong(0);
        WorkerResponseHealthTracker tracker =
                new WorkerResponseHealthTracker(2, 10_000L, false, clock::get);

        Assert.assertFalse(tracker.record(424)); // t=0, count=1
        // Jump well beyond the 10s window: the t=0 error must be pruned, so this does NOT reach the
        // threshold of 2 even though two errors have been recorded in total.
        clock.set(20_000L);
        Assert.assertFalse(tracker.record(424), "stale error must be pruned, so no trip");
        Assert.assertEquals(tracker.windowErrorCount(), 1);

        // A second error while the t=20s error is still inside the window now trips.
        clock.set(20_001L);
        Assert.assertTrue(tracker.record(424));
    }

    @Test
    public void testResetClearsWindow() {
        WorkerResponseHealthTracker tracker = new WorkerResponseHealthTracker(2, 60_000L, false);
        Assert.assertFalse(tracker.record(500));
        Assert.assertTrue(tracker.record(500));
        tracker.reset();
        Assert.assertEquals(tracker.windowErrorCount(), 0);
        Assert.assertFalse(tracker.record(500), "after reset a single error is below threshold");
    }

    @Test
    public void testIsErrorCode() {
        Assert.assertTrue(WorkerResponseHealthTracker.isErrorCode(424));
        Assert.assertTrue(WorkerResponseHealthTracker.isErrorCode(500));
        Assert.assertTrue(WorkerResponseHealthTracker.isErrorCode(507));
        Assert.assertFalse(WorkerResponseHealthTracker.isErrorCode(200));
        Assert.assertFalse(WorkerResponseHealthTracker.isErrorCode(204));
    }
}
