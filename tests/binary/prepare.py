#!/usr/bin/env python
#
# Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import urllib.request
import logging
import os
import argparse

logging.basicConfig(level=logging.INFO)


def download(url, dest):
    logging.info("Downloading " + url)
    urllib.request.urlretrieve(url, dest)


def cleanup(dest):
    if os.path.exists(dest):
        os.remove(dest)


def pt_model(clean=False):
    url = "https://resources.djl.ai/test-models/traced_resnet18.pt"
    dest = "model.pt"
    if clean:
        cleanup(dest)
    else:
        download(url, dest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare ML models')
    parser.add_argument("model", type=str, help="model type like pt, mx, tf")
    parser.add_argument("--clean", dest='clean', action='store_true')
    args = parser.parse_args()
    if args.model == "pt":
        pt_model(args.clean)
    else:
        raise KeyError("Model {} not supported!".format(args.model))
