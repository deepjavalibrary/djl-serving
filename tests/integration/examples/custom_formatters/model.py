import logging
import os
import time

from djl_python import Input
from djl_python.encode_decode import decode
from djl_python.input_parser import input_formatter
from djl_python.output_formatter import TextGenerationOutput, output_formatter
from djl_python.request_io import TextInput

# Import the integration script for custom processing
import integration_script


def get_logger(logger_name):
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger_local = logging.getLogger(logger_name)

    enable_debug = os.getenv("EINSTEIN_DEBUG", "0") == "1"
    if enable_debug:
        logger_local.setLevel(logging.DEBUG)
        logger_local.debug(f"Setting log_level DEBUG for logger: {logger_name}")

    return logger_local


logger = get_logger(__name__)


class MyRequestInput(TextInput):
    trace_id: str
    input_payload: dict
    _custom_processing_params: dict  # options as par the user request to process the input/output


@input_formatter
def custom_input_formatter(decoded_payload, tokenizer=None):
    """
    Simple input formatter for vLLM that transforms custom payload format.
    """
    return integration_script.pre_process(decoded_payload, tokenizer)


@output_formatter
def custom_output_formatter(response):
    """
    Custom output formatter that works on raw vLLM response.
    Handles both streaming and non-streaming responses.
    """
    import json
    import time
    
    timestamp = int(time.time())
    logger.info(f"CUSTOM_OUTPUT_FORMATTER_CALLED at {timestamp}")
    
    # Work with vLLM response objects
    if hasattr(response, 'choices') and response.choices:
        try:
            # Get the first choice
            first_choice = response.choices[0]
            
            # Extract text from different response types
            if hasattr(first_choice, 'delta') and hasattr(first_choice.delta, 'content'):
                # Streaming response chunk
                if first_choice.delta.content:
                    # Only process non-empty content chunks
                    first_choice.delta.content = f"[CUSTOM_FORMATTED:{timestamp}] {first_choice.delta.content}"
            elif hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
                # Chat completion response
                generated_text = first_choice.message.content
                result = integration_script.post_process({'decoded_text': generated_text})
                
                try:
                    result_dict = json.loads(result)
                    result_dict['custom_formatter_applied'] = True
                    result_dict['formatter_timestamp'] = timestamp
                    first_choice.message.content = json.dumps(result_dict)
                except json.JSONDecodeError:
                    custom_result = {
                        'custom_formatter_applied': True,
                        'formatter_timestamp': timestamp,
                        'processed_result': result
                    }
                    first_choice.message.content = json.dumps(custom_result)
                    
            elif hasattr(first_choice, 'text'):
                # Text completion response
                generated_text = first_choice.text
                result = integration_script.post_process({'decoded_text': generated_text})
                
                try:
                    result_dict = json.loads(result)
                    result_dict['custom_formatter_applied'] = True
                    result_dict['formatter_timestamp'] = timestamp
                    first_choice.text = json.dumps(result_dict)
                except json.JSONDecodeError:
                    custom_result = {
                        'custom_formatter_applied': True,
                        'formatter_timestamp': timestamp,
                        'processed_result': result
                    }
                    first_choice.text = json.dumps(custom_result)
                
        except Exception as e:
            logger.error(f"Error in custom_output_formatter: {e}")
    
    return response
