import logging
import os

from vllm.worker.neuron_llama3_mm_runner import trace, load_neuron_model

from djl_python.neuron_utils.model_loader import ModelLoader
from djl_python.neuron_utils.utils import get_generation_config


class NxDModelLoader(ModelLoader):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = None
        self.generation_config = get_generation_config(
            model_id_or_path=self.config.model_id_or_path,
            load_path=None
        )
        self.compiled_graph_path = None

        # self.model_loader = InferenceRunner(
        #     model_path=self.config.model_id_or_path,
        #     tokenizer=kwargs.get("tokenizer"),
        #     generation_config=self.generation_config,
        # )

    def load_model(self, **kwargs):
        # TODO TNX: Determine whether partition is required or not

        # Compile only if necessary
        logging.info(f"LLM sharding and compiling for NxD started...")
        self.compiled_graph_path = os.path.join(self.get_load_path(),
                                                "compiled")

        self.partition(self.compiled_graph_path, **kwargs)
        # load the model
        self.model = load_neuron_model(self.compiled_graph_path)
        return self.model

    def partition(self, save_path, **kwargs):
        logging.info("Compiling model to NeuronX Distributed format")

        # TODO: change max_prompt_length and sequence_length
        # TODO: Figure out what are the other kwargs
        trace(
            traced_model_path=save_path,
            model_path=self.config.model_id_or_path,
            hf_config=None,
            tokenizer=None,
            neuron_config=None,
            tp_degree=self.config.tensor_parallel_degree,
            batch_size=self.config.max_rolling_batch_size,
            max_prompt_length=self.config.max_prompt_length,
            sequence_length=self.config.sequence_length,
            enable_bucketing=True,
            **kwargs
        )
