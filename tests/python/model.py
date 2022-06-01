from djl_python import Input, Output


def handle(inputs: Input) -> None:
    if inputs.is_empty():
        return None
    content_type = inputs.get_property("content-type")
    if content_type == "tensor/ndlist":
        return Output().add_as_numpy(np_list=inputs.get_as_numpy())
    elif content_type == "tensor/npz":
        return Output().add_as_npz(np_list=inputs.get_as_npz())
    elif content_type == "application/json":
        return Output().add_as_json(inputs.get_as_json())
    elif content_type is not None and content_type.startswith("text/"):
        return Output().add(inputs.get_as_string())
    else:
        return Output().add(inputs.get_as_bytes())
