from djl_python.input_parser import input_formatter
from djl_python.output_formatter import output_formatter

@output_formatter
def custom_output_formatter(response_data):
    raise RuntimeError("Output formatter intentionally failed")