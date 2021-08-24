from protocol.output import Output
from protocol.request import Request
from util.codec_utils import decode_input
from util.numpy_djl_util import np_to_djl_encode
from util.packaging_util import get_class_name


def _exec_processor(request: Request, function_param):
    python_file = request.get_python_file()
    function_name = request.get_function_name()

    processor_class = get_class_name(python_file, function_name)
    preprocessor = processor_class()
    data = getattr(preprocessor, function_name)(function_param)
    return data


def run_processor(request: Request) -> Output:
    input_bytes = request.get_function_param()
    input = decode_input(input_bytes)
    np_list = _exec_processor(request, input)
    djl_bytes = np_to_djl_encode(np_list)
    return djl_bytes
