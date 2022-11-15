# DJL Serving Inference

## Overview

DJL Serving Inference refers to the process of loading a model into memory with DJL Serving in order to make predictions based on input data.

Predictions API:

`POST /predictions/{model_name}`

`POST /predictions/{model_name}/{version}`

Note: Including `{version}` is optional. If omitted, the latest version of the model is used.

You can run inference using:

1. [UI](#ui)
2. [Curl](#curl)
3. [Postman](#postman)
4. [Python](#python)
5. [Java](#java)

## UI

See the [Model Inference](console.md#model-inference) section of the DJL Serving Console to see how to make inference requests using UI.

## Curl

We can use the curl tool to send a predict request as a POST to DJL Serving's REST endpoint.

In the first example, let's load an Image Classification model and make predictions. An Image Classification model generally takes images as input and output a list of categories with probabilities.

```
# Register model
curl -X POST "http://localhost:8080/models?url=https://resources.djl.ai/demo/pytorch/traced_resnet18.zip&engine=PyTorch"

# Run inference
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST http://localhost:8080/predictions/traced_resnet18 -T kitten.jpg
```

The above command specifies the image as binary input. Or you can use multipart/form-data format as below:

```
curl -X POST http://localhost:8080/predictions/traced_resnet18 -F "data=@kitten.jpg"
```

This should return the following result:

```json
[
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.4021684527397156
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.2915370762348175
  },
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.27031460404396057
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.007626926526427269
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.004957367666065693
  }
]
```

In the second example, we can load a HuggingFace Bert QA model and make predictions.

```
# Register model
curl -X POST "http://localhost:8080/models?url=https://mlrepo.djl.ai/model/nlp/question_answer/ai/djl/huggingface/pytorch/deepset/bert-base-cased-squad2/0.0.1/bert-base-cased-squad2.zip&engine=PyTorch"

# Run inference
curl -k -X POST http://localhost:8080/predictions/bert_base_cased_squad2 -H "Content-Type: application/json" \
    -d '{"question": "How is the weather", "paragraph": "The weather is nice, it is beautiful day"}'
```

The above curl command passes the data to the server using the Content-Type application/json.

This should return the following result:

```
nice
```

In the third example, we can try a HuggingFace Fill Mask model. Masked model inputs masked words in a sentence and predicts which words should replace those masks.

```
# Register model
curl -X POST "http://localhost:8080/models?url=https://mlrepo.djl.ai/model/nlp/fill_mask/ai/djl/huggingface/pytorch/bert-base-uncased/0.0.1/bert-base-uncased.zip&engine=PyTorch"

# Run inference
curl -X POST http://localhost:8080/predictions/bert_base_uncased -H "Content-Type: application/json" -d '{"data": "The man worked as a [MASK]."}'
```

The above curl command passes the data to the server using the Content-Type application/json.

This should return the following result:

```json
[
  {
    "className": "carpenter",
    "probability": 0.05010193586349487
  },
  {
    "className": "salesman",
    "probability": 0.027945348992943764
  },
  {
    "className": "mechanic",
    "probability": 0.02747158892452717
  },
  {
    "className": "cop",
    "probability": 0.02429874986410141
  },
  {
    "className": "contractor",
    "probability": 0.024287723004817963
  }
]
```

## Postman

We can also send predict requests in [Postman](https://www.postman.com/) REST Client app.

Refer [here](https://github.com/deepjavalibrary/djl-demo/tree/master/djl-serving/postman-client) to see how to make inference requests using Postman.

## Python

In Python, we'll use the [requests](https://pypi.org/project/requests/) library's POST function to post data via HTTP.

Refer [here](https://github.com/deepjavalibrary/djl-demo/tree/master/djl-serving/python-client) to see how to make inference requests using Python.

## Java

In Java, we'll use the HttpClient to make POST requests to load model and drive model inference.

Refer [here](https://github.com/deepjavalibrary/djl-demo/tree/master/djl-serving/java-client) to see how to make inference requests using Java.
