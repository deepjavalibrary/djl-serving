# Large model inference

DJLServing has the capability to host large language models and foundation models that does not fit into a single GPU. We maintain a collection of deep learning containers (DLC) specifically designed for conducting inferences with large models, and you can explore the available deep learning containers [here](https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers). The [AWS DLC for LMI](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference-dlc.html) provides documentation detailing the description of libraries available for use with these DLCs.

## LMI containers Configurations 

Beyond DJLServing configurations, the capability for large model inference involves additional settings. The [LMI configuration](lmi/configurations_large_model_inference_containers.md) document organizes these configurations based on the engines present in our DLCs.

These configurations can be specified in two manners: firstly, through the serving.properties file, and secondly, via environment variables within the Docker environment. For a comprehensive guide on specifying these configurations, refer to the [LMI environment variable instruction](lmi/lmi_environment_variable_instruction.md) document, which offers detailed instructions.

## Tutorials

* [TensorRT-LLM ahead of time compilation of models tutorial](lmi/tutorials/trtllm_aot_tutorial.md)
* [TensorRT-LLM manual compilation of models tutorial](lmi/tutorials/trtllm_manual_convert_tutorial.md)
* [HuggingFace Accelerate scheduler tutorial](lmi/tutorials/seq_scheduler_tutorial.md)

## Tuning guides

Depending on your model architecture, model size, and the instance type in use, adjustments to certain configurations may be necessary to optimize instance resource utilization without encountering Out of Memory (OOM) issues. These documents below provides recommended configurations for popular models, tailored to the specific library you are utilizing, such as DeepSpeed, TensorRT, Transformers-NeuronX or LMI Dist.

* [LMI Dist tuning guide](lmi/tuning_guides/lmi_dist_tuning_guide.md)
* [TensorRT-LLM tuning guide](lmi/tuning_guides/trtllm_tuning_guide.md)
* [Transformers-NeuronX tuning guide](lmi/tuning_guides/tnx_tuning_guide.md)
* [DeepSpeed tuning guide](lmi/tuning_guides/deepspeed_tuning_guide.md)

## SageMaker LMI containers resources

* [SageMaker sample notebooks for LLM](https://github.com/nd7141/djl-demo/tree/master/aws/sagemaker/large-model-inference#readme)
* [AWS SageMaker docs on LMI](https://docs.aws.amazon.com/sagemaker/latest/dg/large-model-inference.html)

