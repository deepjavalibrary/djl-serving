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

import copy
from collections import defaultdict
from djl_python import __version__
import logging


# https://docs.aws.amazon.com/deep-learning-containers/latest/devguide/logging-and-monitoring.html
class SMLogFilter(logging.Filter):
    sm_log_markers = [
        'ModelServerError', 'UserScriptError', 'SysHealth',
        'ModelServerTelemetry'
    ]
    counter = defaultdict(int)

    def filter(self, record):
        try:
            if isinstance(record.msg, str):
                for i in self.sm_log_markers:
                    if record.msg.startswith(i + ':'):
                        altered_record = copy.deepcopy(record)
                        tag, metric_name, metric = [
                            i.strip() for i in altered_record.msg.split(':')
                        ]
                        value, units = metric.split(' ')
                        altered_metric_name = ''.join([
                            word[0].upper() + word[1:]
                            for word in metric_name.split(' ')
                        ])
                        altered_record.msg = f"{tag}.Count:{self.count(altered_metric_name)}|#DJLServing:{__version__},{altered_metric_name}:{value} {units}"
                        return altered_record
                return False
            else:
                return False
        except Exception as exc:
            logging.warning(
                f"Forwarding {str(record)} failed due to {str(exc)}")
            return False

    def count(self, key):
        self.counter[key] += 1
        return self.counter[key]
