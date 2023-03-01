import fastertransformer as ft
from djl_python import Input, Output
import logging
from typing import Optional

class FasterTransformerService(object):
    def __init__(self) -> None:
        self.initialized = False

    def initialize(self, properties):
        self.tensor_parallel_degree = int(properties.get("tensor_parallel_degree", 1))
        self.pipeline_parallel_degree = int(properties.get("pipeline_parallel_degree", 1))
        self.dtype = properties.get("dtype", "fp32")
        self.model_id = properties.get("model_id")
        self.model = self.load_model()
        self.initialized = True

    def load_model(self):
        logging.info(f"Loading model: {self.model_id}")
        return ft.init_inference(self.model_id, self.tensor_parallel_degree,
                                    self.pipeline_parallel_degree, self.dtype)

    def inference(self, inputs: Input):
        try:
            # TODO: Add support for more content types
            data = inputs.get_as_json()
            input_text = data["text"]
            if isinstance(input_text, str):
                input_text = [input_text]
            result = self.model.pipeline_generate(input_text)
            outputs = Output().add(result)
        except Exception as e:
            logging.exception("FasterTransformer inference failed")
            outputs = Output().error((str(e)))

        return outputs


_service = FasterTransformerService()

def handle(inputs: Input) -> Optional[Output]:
    
    if not _service.initialized:
        _service.initialize(inputs.get_properties())
    
    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None
    
    return _service.inference(inputs)