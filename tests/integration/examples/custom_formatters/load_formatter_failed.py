from djl_python.input_parser import input_formatter
from djl_python.output_formatter import output_formatter

@input_formatter
def custom_input_formatter(decoded_payload, **kwargs):
    invalid syntax here to cause load error