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

import logging
import os
from abc import ABC, abstractmethod

from djl_python.inputs import Input
from djl_python.outputs import Output

logger = logging.getLogger(__name__)


class AdapterManagerMixin(ABC):
    """
    Mixin class that provides adapter management functionality.
    Services can inherit from this to get adapter registration, update, and deletion capabilities.
    
    This mixin focuses solely on adapter weight management (loading/unloading adapters).
    Custom code management is handled by AdapterFormatterMixin.
    """

    def __init__(self):
        """Initialize adapter management attributes"""
        self.adapter_registry = {}

    @abstractmethod
    async def add_lora(self, lora_name: str, lora_alias: str, lora_path: str):
        """
        Add a LoRA adapter to the engine.
        Must be implemented by the service.
        """
        pass

    @abstractmethod
    async def remove_lora(self, lora_name: str, lora_alias: str):
        """
        Remove a LoRA adapter from the engine.
        Must be implemented by the service.
        """
        pass

    @abstractmethod
    async def pin_lora(self, lora_name: str, lora_alias: str):
        """
        Pin a LoRA adapter in memory.
        Must be implemented by the service.
        """
        pass

    async def register_adapter(self, inputs: Input) -> Output:
        """
        Register a LoRA adapter with the model.
        
        Args:
            inputs: Input containing adapter registration parameters
            
        Returns:
            Output with registration result
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
                raise ValueError(
                    "Can not set preload to false and pin to true")

            # Check if adapter has custom code and mark it
            model_py_path = os.path.join(adapter_path, "model.py")
            inputs.properties["has_custom_code"] = "true" if os.path.isfile(
                model_py_path) else "false"

            # Load adapter weights
            if adapter_preload:
                loaded = await self.add_lora(adapter_name, adapter_alias,
                                             adapter_path)

            if adapter_pin:
                await self.pin_lora(adapter_name, adapter_alias)

            self.adapter_registry[adapter_name] = inputs

        except Exception as e:
            logger.debug(f"Failed to register adapter: {e}", exc_info=True)
            if loaded:
                logger.info(
                    f"LoRA adapter {adapter_alias} was successfully loaded, but failed to pin, unloading ..."
                )
                await self.remove_lora(adapter_name, adapter_alias)
            if any(msg in str(e)
                   for msg in ("No free lora slots",
                               "greater than the number of GPU LoRA slots")):
                raise MemoryError(str(e))
            err = {"data": "", "last": True, "code": 424, "error": str(e)}
            outputs.add(Output.binary_encode(err), key="data")
            return outputs

        logger.info(
            f"Registered adapter {adapter_alias} from {adapter_path} successfully"
        )
        result = {"data": f"Adapter {adapter_alias} registered"}
        outputs.add(Output.binary_encode(result), key="data")
        return outputs

    async def update_adapter(self, inputs: Input) -> Output:
        """
        Update a LoRA adapter.
        
        Args:
            inputs: Input containing adapter update parameters
            
        Returns:
            Output with update result
        """
        adapter_name = inputs.get_property("name")
        adapter_alias = inputs.get_property("alias") or adapter_name
        adapter_path = inputs.get_property("src")
        adapter_preload = inputs.get_as_string("preload").lower(
        ) == "true" if inputs.contains_key("preload") else True
        adapter_pin = inputs.get_as_string(
            "pin").lower() == "true" if inputs.contains_key("pin") else False

        if adapter_name not in self.adapter_registry:
            raise ValueError(f"Adapter {adapter_alias} not registered.")

        outputs = Output()
        try:
            if not adapter_preload and adapter_pin:
                raise ValueError("Can not set load to false and pin to true")

            old_adapter = self.adapter_registry[adapter_name]
            old_adapter_path = old_adapter.get_property("src")
            if adapter_path != old_adapter_path:
                raise NotImplementedError(
                    f"Updating adapter path is not supported.")

            old_adapter_preload = old_adapter.get_as_string("preload").lower(
            ) == "true" if old_adapter.contains_key("preload") else True
            if adapter_preload != old_adapter_preload:
                if adapter_preload:
                    await self.add_lora(adapter_name, adapter_alias,
                                        adapter_path)
                else:
                    await self.remove_lora(adapter_name, adapter_alias)

            old_adapter_pin = old_adapter.get_as_string("pin").lower(
            ) == "true" if old_adapter.contains_key("pin") else False
            if adapter_pin != old_adapter_pin:
                if adapter_pin:
                    await self.pin_lora(adapter_name, adapter_alias)
                else:
                    raise ValueError(
                        f"Unpinning adapter is not supported. To unpin adapter '{adapter_alias}', please delete the adapter and re-register it without pinning."
                    )

            self.adapter_registry[adapter_name] = inputs

        except Exception as e:
            logger.debug(f"Failed to update adapter: {e}", exc_info=True)
            if any(msg in str(e)
                   for msg in ("No free lora slots",
                               "greater than the number of GPU LoRA slots")):
                raise MemoryError(str(e))
            err = {"data": "", "last": True, "code": 424, "error": str(e)}
            outputs.add(Output.binary_encode(err), key="data")
            return outputs

        logger.info(f"Updated adapter {adapter_alias} successfully")
        result = {"data": f"Adapter {adapter_alias} updated"}
        outputs.add(Output.binary_encode(result), key="data")
        return outputs

    async def unregister_adapter(self, inputs: Input) -> Output:
        """
        Unregister a LoRA adapter from the model.
        
        Args:
            inputs: Input containing adapter unregistration parameters
            
        Returns:
            Output with unregistration result
        """
        adapter_name = inputs.get_property("name")
        adapter_alias = inputs.get_property("alias") or adapter_name

        if adapter_name not in self.adapter_registry:
            raise ValueError(f"Adapter {adapter_alias} not registered.")

        outputs = Output()
        try:
            # Remove adapter weights
            await self.remove_lora(adapter_name, adapter_alias)

            # Remove from registry
            del self.adapter_registry[adapter_name]

        except Exception as e:
            logger.debug(f"Failed to unregister adapter: {e}", exc_info=True)
            err = {"data": "", "last": True, "code": 424, "error": str(e)}
            outputs.add(Output.binary_encode(err), key="data")
            return outputs

        logger.info(f"Unregistered adapter {adapter_alias} successfully")
        result = {"data": f"Adapter {adapter_alias} unregistered"}
        outputs.add(Output.binary_encode(result), key="data")
        return outputs
