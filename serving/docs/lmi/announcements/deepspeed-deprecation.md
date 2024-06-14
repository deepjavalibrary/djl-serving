# DeepSpeed deprecation announcement

**Note: This only impacts users that are using the `deepspeed` container that are also using the `DeepSpeed` library for inference.
If you are using lmi-dist, vLLM, or HuggingFace accelerate in this container, you are not impacted.**

With LMI V10 (0.28.0), we are changing the name from LMI DeepSpeed DLC to LMI (LargeModelInference) DLC. 
The `deepspeed` container has been renamed to the `lmi` container.
As part of this change, we have decided to discontinue integration with the DeepSpeed inference library. 
You can continue to use vLLM or LMi-dist Library with the LMI container. If you plan to use DeepSpeed Library, please follow the steps below, or use LMI V9 (0.27.0).

## Fetching the container from SageMaker Python SDK

As part of changing the container name, we have updated the framework tag in the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

To fetch the new image uri from the SageMaker Python SDK:

```python
from sagemaker import image_uris

# New Usage: For the 0.28.0 and future containers
inference_image_uri = image_uris.retrieve(framework="djl-lmi", version="0.28.0", region=region)

# Old Usage: For the 0.27.0 and previous containers
inference_image_uri = image_uris.retrieve(framework="djl-deepspeed", version="0.27.0", region=region)
```

If you have been using the vllm or lmi-dist inference engine, this is the only change you need to make when using the SageMaker Python SDK.
If you have been using the deepspeed inference engine, continue reading for further migration steps. 

## Migrating from DeepSpeed

If you are not using DeepSpeed library (through importing) or using DeepSpeed as your inference engine, you can stop reading from here.
You are leveraging DeepSpeed in the LMI container if you meet any of the following conditions:

You are using the DeepSpeed engine via:

* `engine=DeepSpeed` (serving.properties)
* `OPTION_ENGINE=DeepSpeed` (environment variable)

You are using the built-in DeepSpeed inference handler:

* `option.entryPoint=djl_python.deepspeed` (serving.properties)
* `OPTION_ENTRYPOINT=djl_pytyhon.deepspeed` (environment variable)

You are importing `deepspeed` in a custom inference handler (model.py)

```python
import deepspeed
```

If none of these conditions apply to your use-case, you can stop reading as you are not impacted.

### Custom inference script (model.py)

If you are using a custom handler (model.py) that is not coming from LMI and you import DeepSpeed like this:

```python
import deepspeed
```

You will need to install DeepSpeed manually into our container. Add `deepspeed` to your `requirements.txt` and the model server will install the dependency at runtime. 
In a network isolated deployment environment, you will need to package the `deepspeed` wheel along with other necessary dependencies with the model.py.

### Built-In Handler 

If you set any of the following configurations:

serving.properties file

```
engine=DeepSpeed
```

```
option.entryPoint=djl_python.deepspeed
```

environment variables

``` 
OPTION_ENGINE=DeepSpeed
```

```
OPTION_ENTRYPOINT=djl_python.deepspeed
```

Then you should migrate to:

serving.properties

```
engine=Python
option.mpi_mode=true
```

environment variables

```
HF_MODEL_ID=<hub model id or s3 uri> (container will auto-detect best engine to use for your model)
```



This updated configuration will leverage either `lmi-dist`, `vllm`, or `huggingface accelerate` depending on your model architecture. 
Most model architectures supported by deepspeed are also supported by `lmi-dist` or `vllm`. These inference libraries should provide increased performance compared to `deepspeed`.

We rename DeepSpeed into more generic as Python + MPI way to better represent the use case. 
In the meantime, we inherit most of the DeepSpeed feature with our LMI-Dist Engine, including all models DeepSpeed supported today. Please refer to [LMI-Dist Engine guide](../user_guides/lmi-dist_user_guide.md) to start with.
