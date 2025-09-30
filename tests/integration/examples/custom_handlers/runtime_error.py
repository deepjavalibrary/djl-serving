from djl_python import Input, Output
from djl_python.encode_decode import decode

async def handle(inputs: Input):
    """Custom handler that raises runtime error"""
    raise RuntimeError("Custom handler intentional failure")
