# DJLServing KServe V2 Protocol

## APIs

### Health:

`GET v2/health/live`

`GET v2/health/ready`

`GET v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/ready`

### Model Metadata:

`GET v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]`

### Inference:

`POST v2/models/${MODEL_NAME}[/versions/${MODEL_VERSION}]/infer`

### Reference from Kserve
See [KServe Requirements](https://github.com/kserve/kserve/blob/master/docs/predict-api/v2/required_api.md)