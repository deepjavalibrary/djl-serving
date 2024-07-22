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


def parse_3p_request(input_map: dict, is_rolling_batch: bool, tokenizer,
                     invoke_type: str):
    _inputs = input_map.pop("prompt")
    _param = {
        "details": True,
        "decoder_input_details": True,
        "temperature": input_map.pop("temperature", 0.5),
        "top_p": input_map.pop("top_p", 0.9),
        "max_new_tokens": input_map.pop("max_gen_len", 512),
        "return_full_text": True
    }
    if _param["temperature"] > 0:
        _param["do_sample"] = True
    if invoke_type == "InvokeEndpointWithResponseStream":
        _param["stream"] = True
        _param["output_formatter"] = "3p_stream"
    else:
        _param["output_formatter"] = "3p"
    return _inputs, _param
