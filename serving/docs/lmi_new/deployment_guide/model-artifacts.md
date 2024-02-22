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
 Users can create these artifacts for model architectures that are supported for JIT compilation following this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_aot_tutorial.md). For model architectures that are not supported by TRT-LLM LMI for JIT compilation, follow this [tutorial](https://github.com/deepjavalibrary/djl-serving/blob/master/serving/docs/lmi/tutorials/trtllm_manual_convert_tutorial.md) to create model artifacts. Users can specify the resulting artifacts path as `OPTION_MODEL_ID` during deployment for faster loading than compared to raw Hugging Face model for TRT-LLM LMI.

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

## Compiled models (TensorRT-LLM, Transformers NeuronX)

We recommend that you precompile models when using TensorRT-LLM or Transformers NeuronX in production to reduce the endpoint startup time.
If HuggingFace Pretrained Model artifacts are provided to these backends, they will just-in-time (JIT) compile the model at runtime before it can be used for inference.
This compilation process increases the endpoint startup time, especially as the model size grows.
Please see the respective compilation guides for steps on how to compile your model for the given framework.
* [TensorRT-LLM Compilation Guide]()
* [Transformers NeuronX Compilation Guide]()

Next: [Instance Type Selection](instance-type-selection.md)