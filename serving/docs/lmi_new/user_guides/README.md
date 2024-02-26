# LMI Backend User Guides

LMI provides backend specific user guides that cover the following topics:

* Model Artifact Structure 
  * All backends support standard HuggingFace Transformers Pretrained artifacts
  * The TensorRT-LLM and Transformer-NeuronX user guides provide information on compiled model artifact structures
* Supported Model Architectures
  * Some Model Architectures can only be deployed using specific backends 
* Quick Start Configurations
  * Starter configurations in both `serving.properties` and environment variable formats to provide an out-of-the-box solution for that backend 
* Quantization Guide
  * If a backend supports quantization, we describe the different options and how to enable them
* Advanced Configurations
  * Configurations that are only available with this backend 

The available backends and their respective user guides are available below:

* [DeepSpeed](deepspeed_user_guide.md)
* [LMI-Dist](lmi-dist_user_guide.md)
* [vLLM](vllm_user_guide.md)
* [TensorRT-LLM](trt_llm_user_guide.md)
* [Transformers-NeuronX](tnx_user_guide.md)
