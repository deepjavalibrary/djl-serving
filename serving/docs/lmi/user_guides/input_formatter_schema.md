# Custom input formatter schema

This document provides the schema for the input formatter, allowing you to create your own custom input formatter.

## Signature of your own input formatter

To write your custom input formatter, follow the annotation and signature below:

### For vLLM and TensorRT-LLM backends:
```python
from djl_python.input_parser import input_formatter

@input_formatter
def my_custom_input_formatter(decoded_payload: dict, tokenizer=None, **kwargs) -> dict:
    # your implementation here
    return decoded_payload
```

### For other backends:
```python
from djl_python import Input
from djl_python.request_io import RequestInput
from djl_python.input_parser import input_formatter

@input_formatter
def my_custom_input_formatter(input_item: Input, **kwargs) -> RequestInput:
    # your implementation here
```

You can write this function in your model.py. You don't need to write the handle function in your entry point Python file. DJLServing will search for the `@input_formatter` annotation and apply the annotated function as the input formatter.

### Input arguments for vLLM and TensorRT-LLM backends:
* `@input_formatter` is the annotation that DJLServing will scan for to identify this as the input formatter. Therefore, you do not need to specify anything in serving.properties.
* `decoded_payload` : This is a dictionary containing the already decoded request payload (e.g., JSON to dict conversion already done).
* `tokenizer` : The tokenizer instance (optional parameter).
* `**kwargs` : Contains additional service attributes and configurations.

**Return argument**: Modified payload dictionary in the expected format for the backend.

### Input arguments for other backends:
* `@input_formatter` is the annotation that DJLServing will scan for to identify this as the input formatter. Therefore, you do not need to specify anything in serving.properties.
* `input_item` : This is an `Input` object from DJLServing, which contains the byte array content of a request. You can decode it into any format you want, such as bytearray to JSON, bytearray to image, etc.
  For decoding, you can use the decode utility function in the djl_python package.
* `**kwargs` : Contains most of the class attributes, TRTLLMService or HuggingFaceService. For example, if you use `trtllm` rolling batch, it includes configurations (from serving.properties), tokenizer, and rolling batch of `TRTLLMService`.

**Return argument**: Users are expected to construct a RequestInput and map their inputs to it. For more details on the RequestInput schema, refer to [this link](./output_formatter_schema.md/#requestoutput-schema).

## Examples

### Example for vLLM and TensorRT-LLM backends:
```python
# model.py
from djl_python.input_parser import input_formatter
from djl_python.encode_decode import decode

@input_formatter
def custom_input_formatter(decoded_payload: dict, tokenizer=None, **kwargs) -> dict:
    """
    Custom input formatter for vLLM/TensorRT-LLM backends.
    Transforms custom payload format to expected format.
    
    Args:
        decoded_payload (dict): Decoded request payload
        tokenizer: Tokenizer instance (optional)
        **kwargs: Additional arguments
        
    Returns:
        (dict): Transformed payload in expected format
    """
    # Example: Transform custom "inputPrompt" field to "inputs"
    if "inputPrompt" in decoded_payload:
        decoded_payload["inputs"] = decoded_payload.pop("inputPrompt")
    
    # Example: Set default parameters
    if "parameters" not in decoded_payload:
        decoded_payload["parameters"] = {}
    
    decoded_payload["parameters"].setdefault("max_new_tokens", 256)
    
    return decoded_payload
```

### Example for other backends:
```python
# model.py
from djl_python import Input
from djl_python.input_parser import input_formatter
from djl_python.request_io import TextInput
from djl_python.encode_decode import decode

class MyInput(TextInput):
    my_own_fields : Any

@input_formatter
def custom_input_formatter(input_item: Input, **kwargs) -> RequestInput:
    """
    Replace this function with your own custom input formatter. 
    
    Args:
      input_item (Input): Input object of a request.
      
    Returns:
      (RequestInput): parsed request input
      
      
    """
    content_type = input_item.get_property("Content-Type")

    input_map = decode(input_item, content_type)

    inputs = input_map.pop("inputs", input_map)
    params = input_map.pop("parameters", {})

    request_input = MyInput()
    request_input.input_text = inputs
    request_input.parameters = params
    request_input.my_own_fields = input_map.pop("my_own_fields", None)

    return request_input
```

In the above example, you can also extend the `RequestInput` or `TextInput` classes as needed for your own fields. Please note that these extra fields can be accessed in your custom output formatter but will not be passed down as inference parameters.
