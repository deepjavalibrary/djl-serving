# Starting DJL Serving

Use the following command to start model server locally:

```sh
djl-serving
```

The model server will be listening on port 8080. You can also load a model for serving on start up:

```sh
djl-serving -m "https://resources.djl.ai/demo/mxnet/resnet18_v1.zip"
```

Open another terminal, and type the following command to test the inference REST API:

```sh
curl -O https://resources.djl.ai/images/kitten.jpg
curl -X POST http://localhost:8080/predictions/resnet18_v1 -T kitten.jpg

or:

curl -X POST http://localhost:8080/predictions/resnet18_v1 -F "data=@kitten.jpg"

[
  {
    "className": "n02123045 tabby, tabby cat",
    "probability": 0.4838452935218811
  },
  {
    "className": "n02123159 tiger cat",
    "probability": 0.20599420368671417
  },
  {
    "className": "n02124075 Egyptian cat",
    "probability": 0.18810515105724335
  },
  {
    "className": "n02123394 Persian cat",
    "probability": 0.06411745399236679
  },
  {
    "className": "n02127052 lynx, catamount",
    "probability": 0.010215568356215954
  }
]
```

### Examples for loading models

```shell
# Load models from the DJL model zoo on startup
djl-serving -m "djl://ai.djl.pytorch/resnet"

# Load version v1 of a PyTorch model on GPU(0) from the local file system
djl-serving -m "resnet:v1:PyTorch:0=file:$HOME/models/pytorch/resnet18/"

# Load a TensorFlow model from TFHub
djl-serving -m "resnet=https://tfhub.dev/tensorflow/resnet_50/classification/1"
```

### Examples for customizing data processing

```shell
# Use the default data processing for a well-known application
djl-serving -m "file:/resnet?application=CV/image_classification"

# Specify a custom data processing with a Translator
djl-serving -m "file:/resnet?translatorFactory=MyFactory"

## Pass parameters for data processing
djl-serving -m "djl://ai.djl.pytorch/resnet?applySoftmax=false"
```

### Using DJL Extensions

```shell
# Load a model from an AWS S3 Bucket
djl-serving -m "s3://djl-ai/demo/resnet/resnet18.zip"

# Load a model from HDFS
djl-serving -m "hdfs://localhost:50070/models/pytorch/resnet18/"

# Use a HuggingFace tokenizer
djl-serving -m "file:/bertqa?transaltorFactory=ai.djl.huggingface.BertQATranslator"
```
