#!/usr/bin/env python
#
# Copyright 2026 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
"""
Test Python model that records Prometheus metrics in the default registry of its worker process,
the way an inference engine does.
"""
import time

from prometheus_client import Counter, Gauge

from djl_python import Input
from djl_python import Output

MODEL = "engine_metrics"

# The worker serves the families whose name starts with "vllm:".
REQUESTS = Counter("vllm:test_requests", "Requests handled by the test model.",
                   ["model_name"])
LAST_BATCH = Gauge("vllm:test_last_batch_size",
                   "Size of the last batch of the test model.", ["model_name"])
# The worker does not serve this one: the name has no engine prefix.
PRIVATE = Counter("test_private_requests",
                  "Requests handled by the test model.")


def block():
    time.sleep(3600)


def fail():
    raise RuntimeError("a metric function of the customer fails")


# Functions of the customer that are called when the registry is read, and that never return or
# raise. The worker does not call a collector that has no engine family, so they cannot take the
# engine families down with them.
Gauge("test_blocking_function", "Never returns.").set_function(block)
Gauge("test_failing_function", "Raises.").set_function(fail)

# A labeled metric has no samples before it is used, so create them now. Then a worker that has
# not handled a request yet still shows up in the answer.
REQUESTS.labels(MODEL)
LAST_BATCH.labels(MODEL).set(0)


def handle(inputs: Input):
    """
    Default handler function
    """
    if inputs.is_empty():
        return None

    batches = inputs.get_batches()
    REQUESTS.labels(MODEL).inc(len(batches))
    LAST_BATCH.labels(MODEL).set(len(batches))
    PRIVATE.inc(len(batches))

    outputs = Output()
    for i, item in enumerate(batches):
        outputs.add(item.get_as_string(), key="data", batch_index=i)
    return outputs
