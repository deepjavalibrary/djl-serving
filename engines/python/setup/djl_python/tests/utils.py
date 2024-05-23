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
import sys
from types import FunctionType

WILDCARD_CLASS_MOCKS = ["optimum.exporters.neuron.model_configs"]


def parameters(params_list, naming=None):
    """
    Parameterization:
        Decorate the unittest class and function as in the example below in order to
        test a list of parameters rather than a single test function for each parameter
    Example:

        @parameterize
        class TestClass(unittest.TestCase):

            @parameters([a, b, c])
            def test_function(params):
                # test test_function with parameters a, b and c
    """

    def decorator(func):
        func._params = params_list
        func._naming = naming
        return func

    return decorator


def parameterized(test_case):
    """Test case class must be decorated with @parametrized for @parameters to work for its methods"""
    for attr in test_case.__dict__.copy().values():
        if not isinstance(attr, FunctionType) or not hasattr(attr, "_params"):
            continue

        no_naming = attr._naming is None

        for i, param in enumerate(attr._params):
            name = attr.__name__ + str(i) if no_naming else attr._naming(param)

            def create_method():
                test_method = attr
                test_param = param

                def new_method(self):
                    return test_method(self, test_param)

                new_method.__doc__ = test_method.__doc__
                return new_method

            setattr(test_case, name, create_method())
        delattr(test_case, attr.__name__)
    return test_case


def mock_import_modules(modules):
    try:
        import torch_neuronx
    except ModuleNotFoundError:
        from unittest.mock import MagicMock

        class Mock(MagicMock):

            @classmethod
            def __getattr__(cls, name):
                return MagicMock()

        for mock_module in modules:
            mock = Mock()
            mock.__name__ = mock_module
            if mock.__name__ in WILDCARD_CLASS_MOCKS:
                mock.__all__ = "none"
            sys.modules[mock_module] = mock
