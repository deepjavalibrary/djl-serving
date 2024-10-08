#!/usr/bin/env python
#
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
import logging
import os
from transformers import Pipeline
from peft import PeftModel

from djl_python.inputs import Input
from djl_python.outputs import Output

from djl_python.properties_manager.properties import is_rolling_batch_enabled


def determine_inference_service(properties: dict):
    rolling_batch = properties.get("rolling_batch", "disable")
    if rolling_batch == "disable":
        from .huggingface_dynamic_batch import HuggingFaceDynamicBatchService
        return HuggingFaceDynamicBatchService()
    else:
        from .huggingface_rolling_batch import HuggingFaceRollingBatchService
        return HuggingFaceRollingBatchService()


_service = None


def register_adapter(inputs: Input):
    """
    Registers lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_path = inputs.get_property("src")
    if not os.path.exists(adapter_path):
        raise ValueError(
            f"Only local LoRA models are supported. {adapter_path} is not a valid path"
        )
    logging.info(f"Registering adapter {adapter_name} from {adapter_path}")
    _service.adapter_registry[adapter_name] = inputs
    if not is_rolling_batch_enabled(_service.hf_configs.rolling_batch):
        if isinstance(_service.model, PeftModel):
            _service.model.load_adapter(adapter_path, adapter_name)
        else:
            _service.model = PeftModel.from_pretrained(_service.model,
                                                       adapter_path,
                                                       adapter_name)

        if isinstance(_service.hf_pipeline, Pipeline):
            _service.hf_pipeline.model = _service.model

        if isinstance(_service.hf_pipeline_unwrapped, Pipeline):
            _service.hf_pipeline_unwrapped.model = _service.model
    return Output()


def unregister_adapter(inputs: Input):
    """
    Unregisters lora adapter from the model.
    """
    adapter_name = inputs.get_property("name")
    logging.info(f"Unregistering adapter {adapter_name}")
    #TODO: delete in vllm engine as well
    del _service.adapter_registry[adapter_name]
    if not is_rolling_batch_enabled(_service.hf_configs.rolling_batch):
        _service.model.base_model.delete_adapter(adapter_name)
    return Output()


def handle(inputs: Input):
    """
    Default handler function
    """
    global _service
    if _service is None or not _service.initialized:
        # stateful model
        properties = inputs.get_properties()
        _service = determine_inference_service(properties)
        _service.initialize(properties)

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
