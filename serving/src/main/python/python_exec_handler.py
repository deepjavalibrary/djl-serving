from protocol.request import Request
from util.packaging_util import get_class_name


def _exec_processor(request: Request):
    python_file = request.get_python_file()
    function_name = request.get_function_name()
    function_param = request.get_function_param()

    processor_class = get_class_name(python_file, function_name)
    preprocessor = processor_class()
    data = getattr(preprocessor, function_name)(function_param)
    return data


def run_processor(request: Request) -> bytearray:
    response_data = _exec_processor(request)
    return response_data
