# Release Notes

Below are the release notes for recent Large Model Inference (LMI) images for use on SageMaker.
For details on historical releases, refer to the [Github Releases page](https://github.com/deepjavalibrary/djl-serving/releases).

## LMI V16 (DJL-Serving 0.34.0)

Meet your brand new image! 💿

#### LMI (vLLM) Image – 9-30-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.34.0-lmi16.0.0-cu128
```
* vLLM version upgraded to `0.10.2`
* Going forward, [async mode](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/vllm_user_guide.md#async-mode-configurations) is the officially recommended configuration for the vLLM handler 
* Async vLLM handler now supports custom [input](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/input_formatter_schema.md) and [output](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/lmi/user_guides/output_formatter_schema.md) formatters 
* Async vLLM handler now supports [multi-adapter](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/adapters.md) (LoRA) serving
* Async vLLM handler now supports session-based [sticky routing](https://github.com/deepjavalibrary/djl-serving/blob/0.34.0-dlc/serving/docs/stateful_sessions.md)

## LMI V15 (DJL-Serving 0.33.0)

#### LMI (vLLM) Image – 4-17-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-lmi15.0.0-cu128
```
* vLLM version upgraded to `0.8.4`
* Llama4 Model Support
* Updated Async Implementation, please see the [vLLM async user guide here](user_guides/vllm_user_guide.md#async-mode-configurations) 

#### TensorRT-LLM Image – 6-24-2025
```
763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.33.0-tensorrtllm0.21.0-cu128
```
* TensorRT-LLM version upgraded to `0.21.0rc1`
