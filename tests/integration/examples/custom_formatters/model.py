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
def custom_output_formatter(output):
    """
    Custom output formatter that modifies the output content structure.
    Works with the PairList content structure properly.
    """
    import json
    import time
    
    timestamp = int(time.time())
    logger.info(f"CUSTOM_OUTPUT_FORMATTER_CALLED at {timestamp}")
    
    # Work with the PairList content structure
    if hasattr(output, 'content') and output.content.size() > 0:
        try:
            # Get the first content item
            first_key = output.content.key_at(0)
            first_value = output.content.value_at(0)
            
            # Convert bytes to string for processing
            if isinstance(first_value, (bytes, bytearray)):
                content_str = first_value.decode('utf-8')
            else:
                content_str = str(first_value)
            
            # Try to parse and modify JSON content
            try:
                parsed = json.loads(content_str)
                
                # Add custom metadata
                if isinstance(parsed, dict):
                    parsed['custom_formatter_applied'] = True
                    parsed['formatter_timestamp'] = timestamp
                    
                    # If it has generated_text, call integration script
                    if 'generated_text' in parsed:
                        result = integration_script.post_process({'decoded_text': parsed['generated_text']})
                        parsed['custom_processed'] = True
                        parsed['original_generated_text'] = parsed['generated_text']
                        if isinstance(result, str):
                            try:
                                result_json = json.loads(result)
                                parsed.update(result_json)
                            except json.JSONDecodeError:
                                parsed['processed_result'] = result
                
                # Update the content with modified JSON
                modified_content = json.dumps(parsed, ensure_ascii=False)
                output.content.set_value_at(0, modified_content.encode('utf-8'))
                
            except json.JSONDecodeError:
                # If not JSON, just add a header
                modified_content = f"<!-- CUSTOM_FORMATTER_APPLIED at {timestamp} -->\n{content_str}"
                output.content.set_value_at(0, modified_content.encode('utf-8'))
                
        except Exception as e:
            logger.error(f"Error in custom_output_formatter: {e}")
    
    return output
