#!/usr/bin/env python
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
class Response(object):
    def __init__(self):
        self.len = None
        self.content = None

    def get_len(self):
        return self.len

    def set_len(self, len):
        self.len = len

    def get_buffer_arr(self):
        return self.content

    def set_buffer_arr(self, content):
        self.content = content
