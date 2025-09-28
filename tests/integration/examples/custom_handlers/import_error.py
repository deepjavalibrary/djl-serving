from djl_python import Input, Output
from non_existent_module import NonExistentClass

async def handle(inputs: Input):
    """Custom handler with import error"""
    obj = NonExistentClass()
    return obj.process(inputs)