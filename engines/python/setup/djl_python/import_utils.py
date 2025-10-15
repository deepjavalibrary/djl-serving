#!/usr/bin/env python
#
# Copyright 2025 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import importlib.util
import importlib.metadata


def _is_package_available(pkg_name: str) -> bool:
    """Check if a package is available"""
    package_exists = importlib.util.find_spec(pkg_name) is not None
    if package_exists:
        try:
            importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
    return package_exists


# SKLearn model persistance libraries
_joblib_available = _is_package_available("joblib")
_cloudpickle_available = _is_package_available("cloudpickle")
_skops_available = _is_package_available("skops")

# XGBoost
_xgboost_available = _is_package_available("xgboost")


def is_joblib_available() -> bool:
    return _joblib_available


def is_cloudpickle_available() -> bool:
    return _cloudpickle_available


def is_skops_available() -> bool:
    return _skops_available


def is_xgboost_available() -> bool:
    return _xgboost_available


joblib = None
if _joblib_available:
    import joblib

cloudpickle = None
if _cloudpickle_available:
    import cloudpickle

skops_io = None
if _skops_available:
    import skops.io as skops_io

xgboost = None
if _xgboost_available:
    import xgboost
