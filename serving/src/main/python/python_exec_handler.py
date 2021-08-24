from protocol.request import Request
from util.codec_utils import decode_input
from util.np_util import np_to_djl_encode, djl_to_np_decode
from util.packaging_util import get_class_name


def _exec_processor(request: Request, function_param):
    python_file = request.get_python_file()
    function_name = request.get_function_name()

    processor_class = get_class_name(python_file, function_name)
    preprocessor = processor_class()
    data = getattr(preprocessor, function_name)(function_param)
    return data


def run_processor(request: Request) -> bytearray:
    request_type = request.get_request_type()
    if request_type == 0:
        input_bytes = request.get_function_param()
        input = decode_input(input_bytes)

        # preprocessor result returns list of numpy array
        pre_processor_res = _exec_processor(request, input)
        return np_to_djl_encode(pre_processor_res)
    elif request_type == 1:
        np_list = djl_to_np_decode(request.get_function_param())

        # post processor result returns list of numpy array
        post_processor_res = _exec_processor(request, np_list)
        return np_to_djl_encode(post_processor_res)
    else:
        raise ValueError("Invalid request type provided")
