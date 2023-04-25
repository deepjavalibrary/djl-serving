# Streaming configuration

We explain various options that can be configured while using response streaming when running in [Python mode](https://github.com/deepjavalibrary/djl-serving/blob/e8b48c2bbeaad8d4e8b6b943769582c46ddc7002/serving/docs/modes.md#python-mode). Response streaming can be enabled in djl-serving by setting `enable_streaming` option in `serving.properties` file. 

E.g `serving.properties` file

```
engine=Python
option.dtype=fp16
option.model_id=stabilityai/stablelm-base-alpha-7b
option.tensor_parallel_degree=1
option.enable_streaming=True
```

Currently, DeepSpeed and Python are the two supported options for engine for streaming. As you can see, `enable_streaming` option has been set to True in the above example `serving.properties` file. Users can either use default handlers in djl-serving or custom model.py handler to stream responses.


## Default handlers

Default handlers facilitates no-code approach in djl-serving. With default handlers, model loading, inference and streaming logic is taken care of by djl-serving. Users will only need to care about sending requests and parsing output response. Currently, DeepSpeed, Hugging Face, transformers-neuronx default handlers support streaming capability.

## Custom model.py handler

djl-serving also supports custom handler that users can provide in a model.py file. Please refer to  [Python mode](https://github.com/deepjavalibrary/djl-serving/blob/e8b48c2bbeaad8d4e8b6b943769582c46ddc7002/serving/docs/modes.md#python-mode) for more details on contents of model.py. To stream responses, following key changes should be made in the handler

* Import StreamingUtils module => `from djl_python.streaming_utils import StreamingUtils`
* In the handler code where you would typically call inference functions like model.generate(), fetch one of the stream generator functions implemented in djl-serving using StreamingUtils.get_stream_generator(ENGINE) method. We currently support `DeepSpeed`, `Accelerate`, `transformers-neuronx`  for ENGINE argument. Stream generators follow the signature - `def stream_generator(model, tokenizer, inputs: List[str], **parameters) -> List[str]:`
* Add stream generator function fetched above to the `Output` object of djl-serving using `add_stream_content()` method.  `add_stream_content()` method of Output object follows the signature `def add_stream_content(stream_generator,  output_formatter=_default_stream_output_formatter):`. djl-serving uses a default output formatter to format model output before sending to the client. User can optionally add their own formatter. Details of output formatting is explained below.


Below code snippet shows key changes that are needed to use djl-serving’s streaming generators.

```
def handle(inputs):
    model = load_model(inputs.get_properties())
    # ....
    # ....
    stream_generator = StreamingUtils.get_stream_generator(ENGINE) 
    outputs = Output()
    outputs.add_stream_content(stream_generator(model, tokenizer, inputs, **parameters))
    return outputs
```

## Output formatting

djl-serving uses chunked encoding in the model server layer and the client receives response as a byte array. To make sense of the byte array, clients need to understand the format in which they receive the data. djl-serving provides a default output formatter to format the output from the model.  Default output formatter accepts the list of strings returned by the stream generator and encodes them in a json format where each iteration will be separated by a new line. Default output formatter with an example is explained.

For example, input contains batch size = 2, i.e two input prompts to the model - `[req1, req2]`, then the output is formatted for each iteration of the model like below
Output for iteration:

iteration 1 - `{output_tokens : [req1_token_text1, req2_token_text1]}\n`

iteration 2 - `{output_tokens : [req1_token_text2, req2_token_text2]}\n`

iteration 3 - `{output_tokens : [req1_token_text3, req2_token_text3]}\n`


Users can also implement their own output formatter and pass it as an argument to `add_stream_content()` method described above in custom model.py handler section. Custom output formatter should follow the signature
`def custom_output_formatter(inputs: List[str]) -> bytearray:`

## Supported model kwargs

Accelerate and DeepSpeed streaming generators currently support following Hugging Face model_kwargs used in model inference which can be passed as parameters in the inference request. 

* max_new_tokens
* repetition_penalty
* top_p
* top_k
* typical_p
* manual_seed

