#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import logging
import time

SPECULATIVE_FREQUENCY_SEC = 30.0


class TelemetryManager:

    def __init__(self):
        self.reset_speculative()

    def record_speculative(self, data):
        self.speculative_acceptance_rate_count = self.speculative_acceptance_rate_count + data[
            "acceptance_history_len"]
        self.speculative_acceptance_rate_total = self.speculative_acceptance_rate_total + data[
            "mean_acceptance"] * data["acceptance_history_len"]
        if time.time(
        ) - self.speculative_sent_time > SPECULATIVE_FREQUENCY_SEC:
            mean_acceptance = 1.0 * self.speculative_acceptance_rate_total / self.speculative_acceptance_rate_count
            logging.info(
                f"ModelServerTelemetry: Speculative Decoding Mean Acceptance: {mean_acceptance} rate"
            )
            self.reset_speculative()

    def reset_speculative(self):
        self.speculative_sent_time = time.time()
        self.speculative_acceptance_rate_count = 0
        self.speculative_acceptance_rate_total = 0.0


telemetry_manager = TelemetryManager()
