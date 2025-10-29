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
import json
import logging
import os
from typing import AsyncGenerator, Callable, Optional, Union

from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.input_parser import SAGEMAKER_ADAPTER_IDENTIFIER_HEADER


def create_non_stream_output(data: Union[str, dict],
                             properties: Optional[dict] = None,
                             error: Optional[str] = None,
                             code: Optional[int] = None) -> Output:
    return _create_output(
        data,
        True,
        "application/json",
        properties=properties,
        error=error,
        code=code,
    )


def create_stream_chunk_output(data: Union[str, dict],
                               last_chunk: bool,
                               error: Optional[str] = None,
                               code: Optional[int] = None) -> Output:
    return _create_output(
        data,
        last_chunk,
        "application/jsonlines",
        error=error,
        code=code,
    )


def _create_output(
    data: Union[str, dict],
    last_chunk: bool,
    content_type: str,
    properties: Optional[dict] = None,
    error: Optional[str] = None,
    code: Optional[int] = None,
) -> Output:
    if isinstance(data, dict):
        data_str = json.dumps(data, ensure_ascii=False)
    else:
        data_str = data
    # Ensure newline for proper jsonlines handling. Extra newlines are fine if already in data_str
    data_str = data_str + '\n'
    response_dict = {
        "data": data_str,
        "last": last_chunk,
    }
    if error:
        response_dict["error"] = error
    if code:
        response_dict["code"] = code
    output = Output()
    output.add_property("Content-Type", content_type)
    if properties:
        for k, v in properties.items():
            output.add_property(k, v)
    output.add(Output.binary_encode(response_dict))
    return output


async def handle_streaming_response(
    response: AsyncGenerator[str, None],
    stream_output_formatter: Callable,
    accumulate_chunks: bool = False,
    **kwargs,
) -> AsyncGenerator[Output, None]:
    """
    This utility provides functionality that converts string outputs from one async generator
    into Output objects that can be handled by the async python engine.

    :param response: AsyncGenerator that produces strings
    :param stream_output_formatter: function that converts strings from the generator into new strings to return to the client.
    :param accumulate_chunks: whether to maintain a history of all chunks received and pass to stream_output_formatter
    :return: AsyncGenerator that produces Output objects that are returned to the model server frontend.
    """
    history = []
    async for chunk in response:
        try:
            if accumulate_chunks:
                data, last, history = stream_output_formatter(chunk,
                                                              history=history,
                                                              **kwargs)
            else:
                data, last = stream_output_formatter(chunk, **kwargs)
        except Exception as e:
            logging.exception("stream_output_formatter failed")
            output = create_stream_chunk_output("",
                                                True,
                                                error=str(e),
                                                code=424)
            yield output
            return

        output = create_stream_chunk_output(data, last)
        yield output
        if last:
            return


def _extract_lora_adapter(raw_request, decoded_payload):
    """
    Get lora adapter name from request headers or payload.
    """
    adapter_name = None

    if SAGEMAKER_ADAPTER_IDENTIFIER_HEADER in raw_request.get_properties():
        adapter_name = raw_request.get_property(
            SAGEMAKER_ADAPTER_IDENTIFIER_HEADER)
        logging.debug(f"Found adapter in headers: {adapter_name}")
    elif "adapters" in decoded_payload:
        adapter_name = decoded_payload.get("adapters")
        logging.debug(f"Found adapter in payload: {adapter_name}")

    return adapter_name


async def register_adapter(inputs: Input, service):
    """
    Registers lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name
    adapter_path = inputs.get_property("src")
    adapter_preload = inputs.get_as_string("preload").lower(
    ) == "true" if inputs.contains_key("preload") else True
    adapter_pin = inputs.get_as_string(
        "pin").lower() == "true" if inputs.contains_key("pin") else False

    outputs = Output()
    loaded = False
    try:
        if not os.path.exists(adapter_path):
            raise ValueError(
                f"Only local LoRA models are supported. {adapter_path} is not a valid path"
            )

        if not adapter_preload and adapter_pin:
            raise ValueError("Can not set preload to false and pin to true")

        if adapter_preload:
            loaded = await service.add_lora(adapter_name, adapter_alias,
                                            adapter_path)

        if adapter_pin:
            await service.pin_lora(adapter_name, adapter_alias)
        service.adapter_registry[adapter_name] = inputs
    except Exception as e:
        logging.debug(f"Failed to register adapter: {e}", exc_info=True)
        if loaded:
            logging.info(
                f"LoRA adapter {adapter_alias} was successfully loaded, but failed to pin, unloading ..."
            )
            await service.remove_lora(adapter_name, adapter_alias)
        if any(msg in str(e)
               for msg in ("No free lora slots",
                           "greater than the number of GPU LoRA slots")):
            raise MemoryError(str(e))
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(
        f"Registered adapter {adapter_alias} from {adapter_path} successfully")
    result = {"data": f"Adapter {adapter_alias} registered"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs


async def update_adapter(inputs: Input, service):
    """
    Updates lora adapter with the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name
    adapter_path = inputs.get_property("src")
    adapter_preload = inputs.get_as_string("preload").lower(
    ) == "true" if inputs.contains_key("preload") else True
    adapter_pin = inputs.get_as_string(
        "pin").lower() == "true" if inputs.contains_key("pin") else False

    if adapter_name not in service.adapter_registry:
        raise ValueError(f"Adapter {adapter_alias} not registered.")

    outputs = Output()
    try:
        if not adapter_preload and adapter_pin:
            raise ValueError("Can not set load to false and pin to true")

        old_adapter = service.adapter_registry[adapter_name]
        old_adapter_path = old_adapter.get_property("src")
        if adapter_path != old_adapter_path:
            raise NotImplementedError(
                f"Updating adapter path is not supported.")

        old_adapter_preload = old_adapter.get_as_string("preload").lower(
        ) == "true" if old_adapter.contains_key("preload") else True
        if adapter_preload != old_adapter_preload:
            if adapter_preload:
                await service.add_lora(adapter_name, adapter_alias,
                                       adapter_path)
            else:
                await service.remove_lora(adapter_name, adapter_alias)

        old_adapter_pin = old_adapter.get_as_string("pin").lower(
        ) == "true" if old_adapter.contains_key("pin") else False
        if adapter_pin != old_adapter_pin:
            if adapter_pin:
                await service.pin_lora(adapter_name, adapter_alias)
            else:
                raise ValueError(
                    f"Unpinning adapter is not supported. To unpin adapter '{adapter_alias}', please delete the adapter and re-register it without pinning."
                )
        service.adapter_registry[adapter_name] = inputs
    except Exception as e:
        logging.debug(f"Failed to update adapter: {e}", exc_info=True)
        if any(msg in str(e)
               for msg in ("No free lora slots",
                           "greater than the number of GPU LoRA slots")):
            raise MemoryError(str(e))
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(f"Updated adapter {adapter_alias} successfully")
    result = {"data": f"Adapter {adapter_alias} updated"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs


async def unregister_adapter(inputs: Input, service):
    """
    Unregisters lora adapter from the model.
    """
    adapter_name = inputs.get_property("name")
    adapter_alias = inputs.get_property("alias") or adapter_name

    if adapter_name not in service.adapter_registry:
        raise ValueError(f"Adapter {adapter_alias} not registered.")

    outputs = Output()
    try:
        await service.remove_lora(adapter_name, adapter_alias)
        del service.adapter_registry[adapter_name]
    except Exception as e:
        logging.debug(f"Failed to unregister adapter: {e}", exc_info=True)
        err = {"data": "", "last": True, "code": 424, "error": str(e)}
        outputs.add(Output.binary_encode(err), key="data")
        return outputs

    logging.info(f"Unregistered adapter {adapter_alias} successfully")
    result = {"data": f"Adapter {adapter_alias} unregistered"}
    outputs.add(Output.binary_encode(result), key="data")
    return outputs
