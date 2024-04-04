# Model Artifacts for LMI

LMI Containers support deploying models with artifacts stored in either the HuggingFace Hub, or AWS S3.
For models stored in the HuggingFace Hub you will need the model_id (e.g. [`meta-llama/Llama-2-13b-hf`](https://huggingface.co/meta-llama/Llama-2-13b-hf)).
For models stored in S3 you will need the S3 uri for the object prefix of your model artifacts (e.g. `s3://my-bucket/my-model-artifacts/`).

Model Artifacts must be in the HuggingFace Transformers pretrained format.

## HuggingFace Transformers Pretrained Format

LMI Containers support loading models saved in the HuggingFace Transformers pretrained format.
This means that the model has been saved using the [`save_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained) method from HuggingFace Transformers, and is loadable using the [`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained) method.
Most open source LLMs available on the HuggingFace Hub have been saved using this format and are compatible with LMI containers.
LMI Containers only support loading HuggingFace models weights from PyTorch (pickle) or [SafeTensor](https://huggingface.co/docs/text-generation-inference/conceptual/safetensors) checkpoints.
Most open source models available on the HuggingFace Hub offer checkpoints in at least one of these formats.

In addition to the model artifacts, we expect that the tokenizer has been saved as part of the model artifacts and is loadable using the [`AutoTokenizer.from_pretrained`](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained) method.

A sample of what the model and tokenizer artifacts looks like is shown below:

```
model/
|- config.json [Required](model configuration file with architecture details)
|- model-000X-of-000Y.safetensors (safetensor checkpoint shard - large models will have multiple checkpoint shards)
|- model.safetensors.index.json (safetensor weight mapping)
|- pytorch_model-000X-of-000Y.bin (PyTorch pickle checkpoint shard - large models will have multiple checkpoint shards)
|- pytorch_model.bin.index.json (PyTorch weight mapping)
|- tokenizer_config.json [Required] (tokenizer configuration)
|- special_tokens_map.json (special token mapping)
|- *modelling.py (custom modelling files)
|- *tokenizer.py (custom tokenzer)
|- tokenizer.json (tokenizer model)
|- tokenizer.model (tokenizer model)
```

Please remember to turn on `option.trust_remote_code=true` or `OPTION_TRUST_REMOTE_CODE=true` if you have customized modelling and/or customized tokenizer.py files.

## TensorRT-LLM(TRT-LLM) LMI model format
 TRT-LLM LMI supports loading models in a custom format that includes compiled TRT-LLM engine files and Hugging Face model config files.
 Users can create these artifacts for model architectures that are supported for JIT compilation following this [tutorial](../tutorials/trtllm_aot_tutorial.md). 
 For model architectures that are not supported by TRT-LLM LMI for JIT compilation, follow this [tutorial](../tutorials/trtllm_manual_convert_tutorial.md) to create model artifacts. Users can specify the resulting artifacts path as `option.model_id` during deployment for faster loading than compared to raw Hugging Face model for TRT-LLM LMI.

 Below directory structure represents an example of TensorRT-LLM LMI model artifacts structure.

```
  trt_llm_model_repo
    └── tensorrt_llm
        ├── 1
        │ ├── trt_llm_model_float16_tp2_rank0.engine # trt-llm engine
        │ ├── trt_llm_model_float16_tp2_rank1.engine # trt-llm engine
        │ ├── config.json # trt-llm config file
        │ └── model.cache
        ├── config.pbtxt # trt-llm triton backend config
        ├── config.json # Below are HuggingFace model config files and may vary per model
        ├── pytorch_model.bin.index.json
        ├── requirements.txt
        ├── special_tokens_map.json
        ├── tokenizer_config.json
        └── tokenizer.model
```

## Neuron Pretrained Model Formats

For pretrained models that will be compiled at runtime, the HuggingFace Transformers pretrained format is preferred.

Model compile time can quickly become an issue for larger models, so compiled models are accepted in the following formats.

### Standard Optimum-Neuron model artifacts 2.16.0 SDK
Under the same folder level, we expect:

- config.json: Store the model architecture, structure information, and neuron compiler configuration
- tokenizer_config.json: Store the tokenizer config information
- modelling files (*.py): If your model has custom modelling or tokenizer files.
  - Please remember to turn on `option.trust_remote_code=true` or `OPTION_TRUST_REMOTE_CODE=true`
- checkpoint directory: Directory containing the split-weights model
  - other files that are needed for split model loading
- compiled directory: Directory containing the `neff` files
- other files that are needed for model loading and inference

A sample of what the model and tokenizer artifacts looks like is shown below:

```
model/
|- checkpoint/ 
|- - pytorch_model.bin/ (directory containing the split model weights)
|- - config.json (model configuration of the model before compilation)
|- compiled/
|- - *.neff (files containing the serialization of the compiled model graph)
|- config.json [Required](model configuration file with architecture details)
|- tokenizer_config.json [Required] (tokenizer configuration)
|- special_tokens_map.json (special token mapping)
|- *modelling.py (custom modelling files)
|- *tokenizer.py (custom tokenzer)
|- tokenizer.json (tokenizer model)
|- tokenizer.model (tokenizer model)
```

### Split Model and Compiled Graph 2.16.0 SDK
Split Model: Under the same folder level, we expect:

- config.json: Store the model architecture, structure information, and neuron compiler configuration
- tokenizer_config.json: Store the tokenizer config information
- modelling files (*.py): If your model has custom modelling or tokenizer files.
  - Please remember to turn on `option.trust_remote_code=true` or `OPTION_TRUST_REMOTE_CODE=true`
- pytorch_model.bin: Directory containing the split-weights model (This is not a typo it is a directory)
  - other files that are needed for split model loading

Compiled Graph: Under the same folder level, we expect:
- The files specifying the compiled graph. This can be `.neff` files, or a dump of the `neff` cache.

A sample of what the model and tokenizer artifacts looks like is shown below:

```
model/
|- pytorch_model.bin/ (directory containing the split model weights)
|- config.json [Required](model configuration file with architecture details)
|- tokenizer_config.json [Required] (tokenizer configuration)
|- special_tokens_map.json (special token mapping)
|- *modelling.py (custom modelling files)
|- *tokenizer.py (custom tokenzer)
|- tokenizer.json (tokenizer model)
|- tokenizer.model (tokenizer model)

compiled/
|- *.neff (files containing the serialization, or dumped NEFF cache, of the compiled model graph)
```

To use this format when loading in LMI there are a few advanced configuration details required. The first is the flag
for loading a split model `option.load_split_model`, which indicates the model has already been split and is ready for
loading on neuron devices. The second is the `option.compiled_graph_path` which allows the user to specify either, 
the `*.neff` files compiled for a serialized model, or to a neuron cache directory containing the compiled graph.
This allows for a workaround for models that do not support serialization, or other advanced use cases.

**Note**: Compiled model artifacts must be compiled under the same compiler version as the container being used, if 
the precompiled models compiler version does not match the image the model will fail to load.

## Storing models in S3

For custom models and production use-cases, we recommend that you store model artifacts in S3.

If you want to use a model available from the huggingface hub, you can download the files locally with `git`:

```shell
# Make sure you have git-lfs installed (https://git-lfs.com)
git lfs install
git clone https://huggingface.co/<namespace>/<model>
```

With the model saved locally (either downloaded from the hub, or your own pretrained/fine-tuned model), upload it to S3:

```shell
# Assuming the model artifacts are stored in a directory called model/
aws s3 cp model s3://my-model-bucket/model/ --recursive
```

When specifying configurations for the LMI container, you can also upload the `serving.properties` file to this directory. See the [configuration](configurations.md) section for more details.

## Compiled models (TensorRT-LLM, Transformers NeuronX)

We recommend that you precompile models when using TensorRT-LLM or Transformers NeuronX in production to reduce the endpoint startup time.
If HuggingFace Pretrained Model artifacts are provided to these backends, they will just-in-time (JIT) compile the model at runtime before it can be used for inference.
This compilation process increases the endpoint startup time, especially as the model size grows.
Please see the respective compilation guides for steps on how to compile your model for the given framework:

* [TensorRT-LLM Compilation Guide](../tutorials/trtllm_aot_tutorial.md)
* [Transformers NeuronX Compilation Guide](../tutorials/tnx_aot_tutorial.md)

Next: [Instance Type Selection](instance-type-selection.md)