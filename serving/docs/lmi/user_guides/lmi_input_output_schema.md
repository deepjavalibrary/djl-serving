# LMI handlers Input/Output schema


Our deep learning LMI containers come with default handlers, each handler supporting different optimization engines such as DeepSpeed, lmi-dist, vLLM and TensorRT-LLM.  These handlers are designed to receive input in a specific format and sends output in a predefined format. This document provides the default schema for the input and output of our prediction/invocation API. Checkout [this page](https://docs.djl.ai/docs/serving/serving/docs/inference_api.html) to our know about our inference APIs offered by DJLServing. Please be aware that this schema is applicable to our latest release v0.26.0.

If you wish to create your own pre-processing and post-processing for our handlers, an example can be found here. [here](https://docs.djl.ai/docs/demos/aws/sagemaker/large-model-inference/sample-llm/rollingbatch_llama_7b_customized_preprocessing.html).


## Input schema for our default rolling batch handlers

```
{
    'inputs' : string/list, 
    'parameters' : {
        DeepSpeedRollingBatchParameters/LmiDistRollingBatchParameters/vLLMRollingBatchParameters/TensorRTLLMRollingBatchParameters,
    } 
}
```

## Output schema for our default handlers

#### When rolling batch is enabled
```
application/json & application/jsonlines

Output schema: {
    'generated_text' : string,
    'details': {
        'finish_reason' : Enum: 'length'/'eos_token'/'stop_sequence',
    },
    'token' : [ 
            Token: {
                'id' : number,
                'text' : string,
                'log_prob': float,
                'special_token' : boolean
        }
   ]
}
```

In the output, details will be generated, only if `details=True` is sent in the input. 

#### When rolling batch is disabled
```
Output schema: {
    'generated_text' : string/list
}
```
When providing inputs following the input schema as a string, the output's generated text will be a string. Alternatively, if the inputs are in the form of a list, the generated text will be returned as a list, with each result corresponding to the input items in the list.

### Common rolling batch input parameters
```
    'do_sample' : boolean (default = False),
    'seed' : integer (default = ramdom value),
    'temperature' : float (default= 1.0),
    'repetition_penalty': float (default= 1.0),
    'top_k' : integer (default = 0), 
    'top_p' : float (default= 1.0),
    'max_new_tokens' : integer (default = 30),
    'details' : boolean (default = false, details only available for rolling batch),
    'return_full_text': boolean (default = false),
```

Note: For TensorRTLLM handler, it also has all the common parameters, but it uses different default values. Kindly check below to know the TensorRT LLM default values. 

Apart from these common parameters, there are other parameters that are specific to each handler. 

### DeepSpeed rolling batch input parameters schema

```
DeepSpeedRollingBatchParameters : {
    'typical_p' : float (default= 1.0), 
    'stop_sequences' : list (default = None),
    'truncate' : integer (default = None),
}
```



### LMI Dist rolling batch input parameters schema

```
LmiDistRollingBatchParameters : {
    'typical_p' : float (default= 1.0),
    'stop_sequences' : list (default = None),
    'truncate' : integer (default = None),
    'ignore_eos_token' : boolean (default = false)
}
```



### vLLM rolling batch input parameters schema

```
vLLMRollingBatchParameters : {
    'stop_sequences' : list,
    'temperature' : float (default= 0),
    'top_k' : integer (default = -1)
    
    'min_p': float (default = 0.0),
    'presence_penalty': float (default = 0.0),
    'frequency_penalty' : float (default = 0.0),
    'num_beams': integer (default = 1), (set this greater than 1 to enable beam search)
    'stop_token_ids': list (default = None),
    'include_stop_str_in_output' : boolean (default = false),
    'ignore_eos_token' : boolean (default = false),
    'logprobs' : int (default = None),
    'prompt_logprobs' : int (default = None),
    'skip_special_tokens': boolean (default = true),
    'spaces_between_special_tokens': boolean (default = true),
}
```

### TensorRTLLM rolling batch input parameters schema

For TensorRTLLM handler, it also has all the common parameters, but it uses different default values. 

```
TensorRTLLMRollingBatchParameters : {
    'temperature' : float (default= 0.8),
    'repetition_penalty' : float (default = None),
    'max_new_tokens' : integer (default = 128),
    'top_k' : integer (default = 5), 
    'top_p' : float (default= 0.85),
    'details' : boolean (default = false),
    'stop' : boolean, 
    'presence_penalty': float,
    'length_penalty' : float, 
    'stop_words_list' : list, 
    'bad_words_list' : list, 
    'min_length' : integer
}
```

For those without default values, they remain optional. If these parameters are not provided, they will be ignored.