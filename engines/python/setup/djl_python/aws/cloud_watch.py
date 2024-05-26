#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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


class CloudWatch:

    def __init__(self):
        try:
            import boto3
            self.client = boto3.client("cloudwatch")
        except ImportError:
            self.client = False

    def post(self, key: str):
        if self.client is None:
            logging.info(f"{key} ...")
        else:
            try:
                self.client.put_metric_data(MetricData=[
                    {
                        "MetricName": key,
                        "Unit": "Count",
                        "Value": 1
                    },
                ],
                                            Namespace="DJLServing_sessions")
            except Exception as e:
                logging.debug(f"Failed post cloudwatch metrics: {e}")
