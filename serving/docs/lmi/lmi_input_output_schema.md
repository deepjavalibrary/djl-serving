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

#### When rolling batch is disabled
```
Output schema: {
    'generated_text' : string/list
}
```
When providing inputs following the input schema as a string, the output's generated text will be a string. Alternatively, if the inputs are in the form of a list, the generated text will be returned as a list, with each result corresponding to the input items in the list.


### DeepSpeed rolling batch input parameters schema

```
DeepSpeedRollingBatchParameters : {
    'temperature' : float (default= 1.0),
    'repetition_penalty': float (default= 1.0),
    'top_k' : integer (default = 0), 
    'top_p' : float (default= 1.0),
    'typical_p' : float (default= 1.0),
    'do_sample' : boolean (default = False), 
    'seed' : integer (default = 0),
    'stop_sequences' : list (default = None),
    'max_new_tokens' : integer (default = 30),
    'details' : boolean (default = false, details only available for rolling batch),
    'truncate' : integer (default = None),
}
```



### LMI Dist rolling batch input parameters schema

```
LmiDistRollingBatchParameters : {
    'temperature' : float (default= 1.0),
    'repetition_penalty': float (default= 1.0),
    'top_k' : integer (default = 0), 
    'top_p' : float (default= 1.0),
    'typical_p' : float (default= 1.0),
    'do_sample' : boolean (default = false), 
    'seed' : integer (default = 0),
    'stop_sequences' : list (default = None),
    'max_new_tokens' : integer (default = 30),
    'details' : boolean (default = false),
    'truncate' : integer (default = None)
}
```



### vLLM rolling batch input parameters schema

```
vLLMRollingBatchParameters : {
    'temperature' : float (default= 1.0),
    'repetition_penalty': float (default= 1.0),
    'top_k' : integer (default = 0), 
    'top_p' : float (default= 1.0),
    'stop' : list,
    'max_new_tokens' : integer (default = 30),
    'details' : boolean (default = false),
    'best_of' : int (default = None),
    'min_p': float (default = 0.0),
    'presence_penalty': float (default = 0.0),
    'frequency_penalty' : float (default = 0.0),
    'use_beam_search': boolean (default = false),
    'stop_token_ids': list (default = None),
    'include_stop_str_in_output' : boolean (default = false),
    'ignore_eos' : boolean (default = false),
    'logprobs' : int (default = None),
    'prompt_logprobs' : int (default = None),
    'skip_special_tokens': boolean (default = true),
    'spaces_between_special_tokens': boolean (default = true),
}
```

### TensorRTLLM rolling batch input parameters schema

```
TensorRTLLMRollingBatchParameters : {
    'temperature' : float (default= 0.8),
    'repetition_penalty' : float (default = None),
    'top_k' : integer (default = 5), 
    'top_p' : float (default= 0.85),
    'seed' : integer (default = None),
    'details' : boolean (default = false),
    'return_log_probs' : boolean,
    'stop' : boolean, 
    'len_penalty' : float, 
    'stop_words_list' : list, 
    'bad_words_list' : list, 
    'min_length' : integer
}
```

For those without default values, they remain optional. If these parameters are not provided, they will be ignored.