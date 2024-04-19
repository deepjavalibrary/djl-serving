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
from djl_python import Output, Input
from djl_python.properties_manager.hf_properties import HuggingFaceProperties
from djl_python.utils import read_model_config, get_tokenizer, parse_input_with_formatter, InputFormatConfigs
from djl_python.rolling_batch.rolling_batch import get_content_type_from_output_formatter

LMI_DIST_ADV_MODEL = {
    "RWForCausalLM",
    "GPTNeoXForCausalLM",
    "T5ForConditionalGeneration",
    "LlamaForCausalLM",
    "FalconForCausalLM",
    "MPTForCausalLM",
    "GPTBigCodeForCausalLM",
}


def get_rolling_batch_class_from_str(rolling_batch_type: str, is_mpi: bool,
                                     model_config):
    if rolling_batch_type == "auto":
        architecture = model_config.architectures[0]
        if architecture in LMI_DIST_ADV_MODEL and is_mpi:
            from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
            return LmiDistRollingBatch
        else:
            from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
            return SchedulerRollingBatch
    elif rolling_batch_type == "scheduler":
        from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
        return SchedulerRollingBatch
    elif rolling_batch_type == "lmi-dist":
        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
        return LmiDistRollingBatch
    elif rolling_batch_type == "vllm":
        from djl_python.rolling_batch.vllm_rolling_batch import VLLMRollingBatch
        return VLLMRollingBatch
    raise ValueError(f"Invalid rolling batch type: {rolling_batch_type}")


class RollingBatchService:

    def __init__(self):
        self.tokenizer = None
        self.rolling_batch = None
        self.model_config = None
        self.peft_config = None
        self.initialized = False
        self.adapters = None
        self.adapter_registry = {}
        self.rb_configs = None
        self.input_format_configs = None

    def initialize(self, properties: dict):
        self.rb_configs = HuggingFaceProperties(**properties)
        self.model_config, self.peft_config = read_model_config(
            self.rb_configs.model_id_or_path,
            self.rb_configs.trust_remote_code, self.rb_configs.revision)
        _rolling_batch_cls = get_rolling_batch_class_from_str(
            self.rb_configs.rolling_batch.value, self.rb_configs.is_mpi,
            self.model_config)
        self.rb_configs.kwargs["model_config"] = self.model_config
        self.rolling_batch = _rolling_batch_cls(properties)
        self.tokenizer = get_tokenizer(self.rb_configs.model_id_or_path,
                                       self.rb_configs.trust_remote_code,
                                       self.rb_configs.revision,
                                       peft_config=self.peft_config)
        self.input_format_configs = InputFormatConfigs(
            is_rolling_batch=True,
            is_adapters_supported=True,
            tokenizer=self.tokenizer,
            output_formatter=self.rb_configs.output_formatter)
        self.initialized = True

    def parse_input(
            self, inputs: Input
    ) -> tuple[list[str], list[int], list[dict], dict, list]:
        parsed_input = parse_input_with_formatter(
            inputs, input_format_configs=self.input_format_configs)

        self.adapters = parsed_input.adapters if parsed_input.found_adapters else None

        return parsed_input.input_data, parsed_input.input_size, parsed_input.parameters, parsed_input.errors, parsed_input.batch

    def inference(self, inputs):
        outputs = Output()

        input_data, input_size, parameters, errors, batch = self.parse_input(
            inputs)
        if len(input_data) == 0:
            for i in range(len(batch)):
                err = errors.get(i)
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            return outputs

        if inputs.get_property("reset_rollingbatch"):
            self.rolling_batch.reset()
        if self.adapters is not None:
            adapter_data = []
            for i, a in enumerate(self.adapters):
                if a is None or a == "":
                    adapter_data.append(None)
                elif a in self.adapter_registry:
                    adapter_data.append(self.adapter_registry[a])
                else:
                    adapter_data.append(None)
                    errors[i] = f"Unknown or invalid adapter {a}"
        else:
            adapter_data = None
        result = self.rolling_batch.inference(input_data,
                                              parameters,
                                              adapters=adapter_data)
        idx = 0
        for i in range(len(batch)):
            err = errors.get(i)
            if err:
                err = {"data": "", "last": True, "code": 424, "error": err}
                outputs.add(Output.binary_encode(err),
                            key="data",
                            batch_index=i)
            else:
                outputs.add(Output.binary_encode(result[idx]),
                            key="data",
                            batch_index=i)
                idx += 1

            formatter = parameters[i].get("output_formatter")
            content_type = get_content_type_from_output_formatter(formatter)
            if content_type is not None:
                outputs.add_property(f"batch_{i}_Content-Type", content_type)

        return outputs
