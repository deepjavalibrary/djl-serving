# Adapters

**Note that this API is experimental and is subject to change. Using it requires the environment variable feature flag `ENABLE_ADAPTERS_PREVIEW`.**

DJL Serving has first class support for adapters.
Adapters are patches or changes that can be made to a model to fine tune it for a particular usage.
The benefit of adapters rather than whole model fine-tuning is that they are often smaller and easier to distribute alongside a base model.
This can allow for multiple adapters used at the same system, and sometimes even in the same batch.

With DJL Serving, it is possible to easily work with adapters.
You can create models that accept adapters, use DJL Serving to manage your available adapters, and call models with adapters.

For a concrete usage, see the [large model inference adapters example notebook](http://docs.djl.ai/docs/demos/aws/sagemaker/large-model-inference/sample-llm/multi_lora_adapter_inference.html).

## Managing Adapters

There are several options to choose between for managing your set of adapters.

### Adapters local directory (Recommended)

The easiest option is to use an adapters local directory.
This is as easy as adding a directory of adapters alongside your model files.
It should contain an overarching adapters directory with an artifact directory for each adapter to add.
This works best for having a manageable set of adapters as they are all loaded on startup.
It can be used in conjunction with services like [Amazon SageMaker Single Model Endpoint](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-single-model.html).

```
ls:
serving.properties
model.py (optional)
model artifacts... (optional) 
adapters/
  adapter1/
    ...
  adapter2/
    ...
```

### Management API

The next option is to manage adapters using the management API.
This can be used in conjunction with our existing management API and supports all standard restful options.
See the [Adapter Management API Documentation](adapters_api.md) for details.
However, this option may be difficult to use inside wrapping systems such as Amazon SageMaker.

```
GET  models/{modelName}/adapters               - List adapters
GET  models/{modelName}/adapters/{adapterName} - Get adapter description
POST models/{modelName}/adapters               - Create adapter
DEL  models/{modelName}/adapters/{adapterName} - Delete adapter
```

### Workflow Adapters

The final option for working with adapters is through the [DJL Serving workflows system](workflows.md).
You can use the adapter `WorkflowFunction` to create and call an adapted version of a model within the workflow.
For the simple model + adapter case, you can also directly use the adapter [workflow template](workflow_templates.md).
With our workflows, multiple workflows sharing models will be de-duplicated.
So, the effect of having multiple adapters can be easily made with having one workflow for each adapter.
This system can be used on [Amazon SageMaker Multi-Model Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/multi-model-endpoints.html).

```
workflow.json:
{
  "name": "adapter1",
  "version": "0.1",
  "models": {
    "m": "src/test/resources/adaptecho"
  },
  "configs": {
    "adapters": {
      "a1": {
        "model": "m",
        "src": "url1"
      }
    }
  },
  "workflow": {
    "out": ["adapter", "a1", "in"]
  }
}
```

## Calling Adapters

When calling a model with an adapter, you are also able to specify which adapter to use.
We currently support several techniques for passing in the adapter data.

There are a few things to keep in mind which choosing a calling technique.

1. If you are using workflow adapters, there is no need to specify an adapter as it is included in the workflow.
   Instead, just call the workflow as normal.
2. Each technique must be implemented by the model handler by parsing the adapter from the Input parameter, content, or body respectively.
   Our built-in implementations support it from all options
3. Some of these techniques may not work in all situations. For example, only the custom attributes strategy will work in Amazon SageMaker as it blocks the other options.


### (Recommended) Adapters parameter Calling

This passes the adapter as part of the requeset body.
It will also work with client-side batching and will allow multiple adapters to be passed with one for each input.
If the adapters is not passed, the base model will be used for inference.
You can also specify the base model for an element in the batch by using the empty string `""`.

```

curl -X POST http://127.0.0.1:8080/invocations \
    -H "Content-Type: application/json" \
    -H "X-Amzn-SageMaker-Target-Model: base-1.tar.gz" \
    -d '{"inputs": ["How is the weather"], "adapters": ["adapter_1"], "parameters": {"max_new_tokens": 25}}'
```

### Input Content (and query parameter)

This passes the adapter through a query parameter.
It is reflected in the Input content.
This will not work in Amazon SageMaker.

```
curl -X POST http://127.0.0.1:8080/invocations?adapter=adapter_1 \
    -H "Content-Type: application/json" \
    -H "X-Amzn-SageMaker-Target-Model: base-1.tar.gz" \
    -d '{"inputs": ["How is the weather"], "parameters": {"max_new_tokens": 25}}'
```

### SageMaker Custom Attributes

This passes the adapter through the use of the SageMaker Custom Attributes header.
It is reflected in the Input properties.
This option will work in Amazon SageMaker.

```
curl -X POST http://127.0.0.1:8080/invocations \
    -H "Content-Type: application/json" \
    -H "X-Amzn-SageMaker-Target-Model: base-1.tar.gz" \
    -H "X-Amzn-SageMaker-Custom-Attributes: adapter=adapter_1"
    -d '{"inputs": ["How is the weather"], "parameters": {"max_new_tokens": 25}}'
```

## Models with Adapters

There are two kinds of LoRA that can be put onto various engines.

* **Merged LoRA** - This will apply the adapter by modifying the base model in place.
  It has zero added latency during execution, but has a cost to apply or unapply the merge.
  It works best for cases with only a few adapters.
  It is best for single adapter batches, but doesn’t support multi-adapter batches
* **Unmerged LoRA** - This will alter the model operators to factor in the adapters without changing the base model.
  It has a higher inference latency for the additional adapter operations.
  However, it does support multi-adapter batches.
  It works best for use cases with large numbers of adapters.

With our default handlers, we currently support unmerged LoRA for CausalLM through the huggingface handler.
Support for other model types and handlers is coming soon.

### Writing Custom Adapter Models

Right now, adapters are only supported through our Python engine and not any of the other DJL engines.
See instructions to get started with writing a [python handler](modes.md#python-mode).

To add support for adapters, you must first add the register and unregister like below.
These can then take the adapters and save the src, pre-download it, cache it in memory, or cache it on an accelerator device.

```python
def register_adapter(inputs: Input):
  name = inputs.get_properties()["name"]
  src = inputs.get_properties()["src"]
  # Do adapter registration tasks
  return Output().add("Successfully registered adapter")

def unregister_adapter(inputs: Input):
  name = inputs.get_properties()["name"]
  # Do adapter unregistration tasks
  return Output().add("Successfully unregistered adapter")


def handle(inputs: Input):
  ...
```

Within the handler, you must parse the adapter from the inputs.
The adapter information can be put in the inputs property, content, or body depending on the technique(s) used.
If desired, you can also parse multiple adapter passing options for greater ease of calling.

From there, you can use those as part of your inference call.
This will depend on the specific python deep learning framework you are using for inference.
For example, you would update the [HF Accelerate](https://huggingface.co/docs/transformers/v4.33.0/en/main_classes/text_generation#transformers.GenerationMixin.generate) by passing in the adapters parameter.