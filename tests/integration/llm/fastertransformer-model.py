from djl_python import Input, Output
import fastertransformer

model = None


def load_model(properties):
    tensor_parallel_degree = properties["tensor_parallel_degree"]
    pipeline_parallel_degree = 1  # TODO: add tests for pp_degree > 1
    model_id = properties.get('model_id') or properties.get('model_dir')
    dtype = properties.get("dtype", "fp32")
    return fastertransformer.init_inference(model_id, tensor_parallel_degree,
                                            pipeline_parallel_degree, dtype)


def partition(inputs: Input):
    properties = inputs.get_properties()
    tensor_parallel_degree = properties["tensor_parallel_degree"]
    pipeline_parallel_degree = 1  # TODO: add tests for pp_degree > 1
    model_id = properties.get('model_id') or properties.get('model_dir')
    dtype = properties.get("dtype", "fp32")
    save_mp_checkpoint_path = properties.get("save_mp_checkpoint_path")

    fastertransformer.save_checkpoint(model_id, tensor_parallel_degree,
                                      pipeline_parallel_degree,
                                      save_mp_checkpoint_path, dtype)


def handle(inputs: Input):
    global model

    if not model:
        model = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    input_json = inputs.get_as_json()
    input_data = input_json.pop("inputs")
    result = model.pipeline_generate(input_data)

    return Output().add(result)
