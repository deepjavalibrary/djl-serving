# LMI Text Embedding User Guide

Text Embedding refers to the process of converting text data into numerical vectors.
These embeddings capture the semantic meaning of the text and can be used for various
tasks such as semantic search and similarity detection.

The inference process involves:

1. **Loading a Model**: Loading a model from local directory, S3, DJL model zoo or from huggingface repository.
2. **Tokenization**: Breaking down the input text into tokens that the model can understand.
3. **Embeddings**: Passing the tokens through the model to produce embeddings. Embedding is a
multi-dimension vector that could be used for RAG or general embedding search.

LMI supports Text Embedding Inference with the following engines:

- OnnxRuntime
- PyTorch
- Rust
- Python

Currently, the OnnxRuntime engine provides the best performance for text embedding in LMI. 

## Quick Start Configurations

You can leverage LMI Text Embedding inference using the following starter configurations:

### DJL model zoo

You can specify the `djl://` url to load a model from the DJL model zoo.

```
HF_MODEL_ID=djl://ai.djl.huggingface.onnxruntime/BAAI/bge-base-en-v1.5
# Optional
OPTION_BATCH_SIZE=32
```

### environment variables

You can specify the `HF_MODEL_ID` environment variable to load a model from HuggingFace hub. DJLServing
will download the model from HuggingFace hub and optimize the model with OnnxRuntime at runtime.

```
OPTION_ENGINE=OnnxRuntime
HF_MODEL_ID=BAAI/bge-base-en-v1.5
# Optional
OPTION_BATCH_SIZE=32
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-2-configuration---environment-variables)
to deploy a model with environment variable configuration on SageMaker.

### serving.properties

```
engine=OnnxRuntime
option.model_id=BAAI/bge-base-en-v1.5
translatorFactory=ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory
# Optional
batch_size=32
```

You can follow [this example](../deployment_guide/deploying-your-endpoint.md#option-1-configuration---servingproperties)
to deploy a model with serving.properties configuration on SageMaker.

## Deploy model to SageMaker

The following code example demonstrates this configuration UX using the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk).

This example will use the [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5) model. 

```python
# Assumes SageMaker Python SDK is installed. For example: "pip install sagemaker"
import sagemaker
from sagemaker import Model, image_uris, serializers, deserializers

# Setup role and sagemaker session
role = sagemaker.get_execution_role()  # execution role for the endpoint
session = sagemaker.session.Session()  # sagemaker session for interacting with different AWS APIs

# Fetch the uri of the LMI container
image_uri = image_uris.retrieve(
    framework="djl-lmi",
    region=session.boto_session.region_name,
    version="0.28.0"
)

# Create the SageMaker Model object.
model_id = "BAAI/bge-base-en-v1.5"

env = {
    "HF_MODEL_ID": model_id,
    "OPTION_ENGINE": "OnnxRuntime",
    "SERVING_MIN_WORKERS": "1", # make sure min and max Workers are equals when deploy model on GPU
    "SERVING_MAX_WORKERS": "1",
}

model = Model(image_uri=image_uri, env=env, role=role)

# Deploy your model to a SageMaker Endpoint and create a Predictor to make inference requests
instance_type = "ml.g4dn.2xlarge"
endpoint_name = sagemaker.utils.name_from_base("lmi-text-embedding")

model.deploy(initial_instance_count=1,
             instance_type=instance_type,
             endpoint_name=endpoint_name,
             )

predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=session,
    serializer=serializers.JSONSerializer(),
    deserializer=deserializers.JSONDeserializer(),
)

# Make an inference request against the endpoint
predictor.predict(
    {"inputs": "What is Deep Learning?"}
)
```

The full notebook is available [here](https://github.com/deepjavalibrary/djl-demo/blob/master/aws/sagemaker/large-model-inference/sample-llm/text_embedding_deploy_bert.ipynb).

## Available Environment Variable Configurations

The following environment variables are exposed as part of the UX:

**HF_MODEL_ID**

This configuration is used to specify the location of your model artifacts.

**HF_REVISION**

If you are using a model from the HuggingFace Hub, this specifies the commit or branch to use when downloading the model.

This is an optional config, and does not have a default value. 

**HF_MODEL_TRUST_REMOTE_CODE**

If the model artifacts contain custom modeling code, you should set this to true after validating the custom code is not malicious.
If you are using a HuggingFace Hub model id, you should also specify `HF_REVISION` to ensure you are using artifacts and code that you have validated.

This is an optional config, and defaults to `False`.

**OPTION_ENGINE**

This option represents the Engine to use, values include `OnnxRuntime`, `PyTorch`, `Rust`, etc.

**OPTION_BATCH_SIZE**

This option represents the dynamic batch size.

This is an optional config, and defaults to `1`.

**SERVING_MIN_WORKERS**

This option represents minimum number of workers.

This is an optional config, and defaults to `1`.

**SERVING_MAX_WORKERS**

This option represents the maximum number of workers.

This is an optional config, and default is `#CPU` for CPU, GPU default is `2`.

When running Text Embedding task on GPU, benchmarking result shows `SERVING_MAX_WORKERS=1` gives better performance.
We recommend to use same value for `SERVING_MIN_WORKERS` and `SERVING_MAX_WORKERS` on GPU to avoid worker scaling overhead.

### Additional Configurations

Additional configurations are available to further tune and customize your deployment.
These configurations are covered as part of the advanced deployment guide [here](../deployment_guide/configurations.md).

## API Schema

### Request Schema

Request Body Fields:

| Field Name   | Field Type                                    | Required | Possible Values                                                                                                                                     |
|--------------|-----------------------------------------------|----------|-----------------------------------------------------------------------------------------------------------------------------------------------------|
| `inputs`     | string, array of strings                      | yes      | example: "What is Deep Learning", ["What is Deep Learning", "How many ways can I peel an orange"]                                                   |

Example request using curl:

```
curl -X POST http://127.0.0.1:8080/invocations \
  -H 'Content-Type: application/json' \
  -d '{"inputs":"What is Deep Learning?"}'
```

### Response Schema

The response is returned as an array.

Example response:

```
[
  0.0059961616,
  -1.0498056,
  0.040412642,
  -0.2044975,
  0.8382087,
  0.38618544,
  0.36715484,
  0.013636998,
  -0.005990674,
  -0.5529305,
  -0.41090673,
  0.10696329,
  -0.12136408,
  1.0979345,
  0.022666162,
  0.77606845,
  1.0382273,
  -0.3466831,
  0.18381083,
  0.729008,
  -0.83670723,
  -1.0255599,
  -0.65942657,
  0.29289168,
  -0.023201486,
  0.037433445,
  -0.7203932,
  0.21155629,
  -0.39795896,
  0.18298008,
  0.47596836,
  -0.14674911,
  0.6118379,
  -0.453512,
  0.26682568,
  0.6894229,
  -0.65042716,
  -0.5810942,
  -0.3695681,
  -0.32711855,
  -0.307704,
  -0.6989487,
  -0.19980608,
  0.80877674,
  -1.3377498,
  0.2968479,
  -1.9953098,
  0.03979389,
  -0.3749918,
  -0.20275205,
  -1.2463598,
  0.84681666,
  0.52935517,
  -0.31514803,
  0.7052601,
  0.2518756,
  0.069923475,
  -1.3327819,
  0.16885656,
  -0.93077576,
  0.85949373,
  0.6133,
  -0.30648738,
  -0.45386475,
  -0.015815632,
  -0.43250006,
  0.39866835,
  0.70893,
  -0.38845858,
  -0.057932496,
  -0.4755477,
  0.48192352,
  -0.34001595,
  -0.08236743,
  -0.8661858,
  -0.18446063,
  0.2131256,
  0.239672,
  1.3575845,
  -0.1360039,
  0.061091293,
  1.0282563,
  0.7029718,
  0.5239024,
  0.07606942,
  0.0104247425,
  -0.31992394,
  -0.07821367,
  -0.74150276,
  0.37592173,
  0.17164698,
  -0.87833697,
  0.9938407,
  0.85141015,
  0.23477103,
  0.12609816,
  0.22988753,
  -0.5844423,
  -0.19332272,
  -0.24335314,
  -0.38531983,
  -0.9747727,
  -0.27821776,
  -0.2433357,
  -0.8007462,
  0.44151047,
  0.8632499,
  -0.23099707,
  0.23974843,
  -0.32271224,
  -0.021940248,
  -0.1395864,
  -0.040762194,
  -0.80075353,
  -0.7114315,
  -0.12845717,
  -0.2704254,
  0.061806552,
  -0.4129424,
  -0.7583218,
  0.56789887,
  0.6335885,
  0.032452866,
  0.3529782,
  -0.5598041,
  0.25341547,
  0.064973444,
  0.30256605,
  0.075141996,
  -0.44579333,
  0.34202874,
  0.2800837,
  -0.13066635,
  0.36010095,
  -0.24404356,
  0.0122952005,
  -0.33419338,
  0.109375805,
  0.42626223,
  0.16283292,
  -0.041986328,
  -0.95607823,
  0.4966297,
  -0.58694136,
  0.8990218,
  -0.25514218,
  -0.13172187,
  -0.07102677,
  -0.31220224,
  0.045226548,
  -0.7334137,
  0.9173523,
  -0.3123617,
  -0.8177161,
  0.13754754,
  0.6586415,
  -0.0068656555,
  0.098955,
  0.56925595,
  0.38343358,
  0.97359407,
  -0.32263,
  -0.90202767,
  -0.16135655,
  -0.40491983,
  -0.89586705,
  0.91414654,
  0.40440056,
  -0.33186355,
  0.41235745,
  -0.036074575,
  -0.1890843,
  0.33652487,
  -0.19793989,
  0.19175956,
  0.7811746,
  0.4763296,
  0.56242734,
  0.71122605,
  0.8491324,
  -1.0836195,
  0.7694596,
  -0.14530765,
  -0.10505987,
  -0.7392691,
  -0.5241517,
  0.82047564,
  -0.18007974,
  0.6071202,
  0.23705445,
  -1.0559548,
  -0.60109943,
  -0.55704814,
  -0.622884,
  0.5492192,
  -0.2796213,
  -0.25005502,
  1.3561565,
  -0.1830543,
  0.7690981,
  0.89704794,
  -0.8499147,
  0.253895,
  -0.27030045,
  -0.45014012,
  0.49377465,
  0.2624732,
  -0.7224904,
  -0.05645481,
  0.057679247,
  0.12322178,
  -0.8145645,
  -0.0875828,
  0.023763582,
  0.7144934,
  0.50256175,
  -0.13208336,
  -0.07967954,
  0.108432345,
  -0.30044904,
  0.0860565,
  -0.69734097,
  0.40590292,
  0.04972869,
  0.13519154,
  1.453644,
  0.6807556,
  -0.8543344,
  0.57886934,
  -0.61681986,
  0.06179464,
  0.40478837,
  -0.2085024,
  0.31706032,
  0.22799756,
  -0.45475867,
  0.24009831,
  0.17121156,
  0.7478102,
  -0.55466986,
  -1.1630594,
  0.11896758,
  -0.7829722,
  0.8751456,
  0.19613068,
  0.15642299,
  0.5501064,
  -0.23134911,
  -0.2423896,
  -0.27698904,
  -0.07504316,
  -0.0011924113,
  0.27453882,
  0.47647652,
  -0.3938967,
  -0.97887135,
  0.032227524,
  -0.16045108,
  1.6983161,
  0.45556778,
  -0.8375664,
  0.8493691,
  -0.45522866,
  0.36748475,
  -0.42189863,
  -1.5736749,
  -0.47000903,
  -0.33056363,
  1.4479512,
  -0.23309126,
  0.79736155,
  0.122715764,
  0.6489947,
  -6.2803236E-5,
  -0.44395,
  0.08917888,
  0.9087269,
  0.9115147,
  -0.20870346,
  -0.71971977,
  -0.4009451,
  1.1004946,
  0.0929569,
  -0.250469,
  0.44864234,
  -0.43092313,
  -0.05047079,
  -0.14335248,
  -0.15065433,
  0.071684726,
  0.59264517,
  1.0204256,
  -0.70037204,
  0.80493677,
  0.34925526,
  0.88331395,
  0.75762135,
  -0.45514092,
  -0.25838882,
  0.002875129,
  -0.116328426,
  0.65513307,
  -0.35742223,
  0.6401003,
  0.3311727,
  -0.21645029,
  1.3131056,
  -0.24388537,
  -3.2505202,
  -0.21489373,
  0.29909623,
  -0.12177677,
  0.6305351,
  -0.9802819,
  -0.096341565,
  -0.16500625,
  0.027056746,
  -0.3810414,
  -0.44811323,
  -0.107583344,
  0.6274576,
  0.41884008,
  0.32803023,
  -0.2725732,
  0.14570288,
  -0.2759351,
  -0.60618335,
  0.75007296,
  -0.5194966,
  -0.48378003,
  -0.20910238,
  1.1802471,
  -0.479603,
  0.45159635,
  -1.3988259,
  -0.8338212,
  -0.050855864,
  -0.037028395,
  -0.39394483,
  -0.47026226,
  -0.53538513,
  0.80559593,
  0.38406372,
  0.26782346,
  0.45964003,
  -0.65812236,
  -0.040072072,
  0.290196,
  -0.5532784,
  -0.8029445,
  0.049611103,
  -0.6748502,
  0.7430752,
  0.26062042,
  -1.5363009,
  -0.7974456,
  -0.24642277,
  0.020854123,
  -0.17664796,
  -0.40674758,
  -1.1205623,
  -0.02085145,
  0.2562925,
  -0.08129512,
  0.042836197,
  -0.19786021,
  -0.41127777,
  0.60389215,
  -0.053745985,
  -0.33683503,
  -0.3768564,
  -0.98018974,
  -0.3432021,
  -0.38665968,
  -1.009802,
  -1.031008,
  0.28705624,
  0.5533665,
  -0.16860572,
  0.7930716,
  -1.0645978,
  -0.80604017,
  0.11406805,
  -0.4248146,
  -0.48853314,
  -0.15658051,
  0.907333,
  0.4493551,
  0.13650355,
  -1.0371721,
  0.58254707,
  0.199945,
  -0.15749091,
  -0.8006224,
  0.76640695,
  -0.6121354,
  -0.33418527,
  -0.47036627,
  1.0668803,
  -0.10671027,
  -0.43982565,
  -0.4343469,
  0.5789606,
  0.3317307,
  0.05429487,
  0.47653046,
  -0.21307655,
  0.32088915,
  -0.34994712,
  -0.40618652,
  -0.72280514,
  -0.27442384,
  -0.6740682,
  -0.36247405,
  -0.16771632,
  0.49512538,
  1.6728407,
  -0.042792857,
  -0.62244827,
  0.23541792,
  0.4193854,
  0.013200714,
  0.0051204683,
  -0.508162,
  0.26859176,
  0.024494493,
  0.30846822,
  -0.6374055,
  -0.5290271,
  0.9262641,
  0.09125612,
  0.74562734,
  0.2532722,
  -0.4898072,
  0.016409349,
  0.2515397,
  0.3151341,
  -0.42605487,
  0.34448025,
  -0.0057288194,
  0.20653899,
  0.14861502,
  -0.0253694,
  -0.21174066,
  -0.18371966,
  -0.11563054,
  -0.15892655,
  0.15574731,
  -0.12771265,
  0.027742216,
  0.27574277,
  -0.2625948,
  -0.34240508,
  0.8183645,
  -1.2124608,
  1.0034169,
  -0.11373905,
  -0.2432877,
  0.24376862,
  -0.47639894,
  0.41671154,
  -0.9653815,
  0.4970341,
  -0.11750264,
  0.16793783,
  -0.108437315,
  -0.47287416,
  -0.7300559,
  0.1645548,
  0.49839726,
  -0.16526854,
  0.17274109,
  0.92254215,
  0.8987754,
  -0.4362364,
  -1.0188268,
  0.5145923,
  0.20084324,
  0.5625598,
  -0.48971194,
  -0.14288938,
  -0.52283686,
  -0.24525356,
  0.6068987,
  0.43570745,
  -0.09429612,
  0.41335583,
  0.9717539,
  0.2860962,
  0.64698756,
  -0.05734274,
  0.54631644,
  0.8332974,
  0.54136103,
  -0.4902781,
  0.09807518,
  -0.9091184,
  0.4623836,
  -0.63153464,
  0.40226787,
  -0.28529564,
  -0.3228058,
  -0.32955492,
  -0.4590344,
  -0.3084508,
  0.7776922,
  -0.055433553,
  -0.34131756,
  -0.656627,
  -0.31522712,
  -0.43529567,
  -0.38323513,
  0.8198578,
  -1.2882359,
  0.50641626,
  0.80444396,
  0.7380471,
  0.60722286,
  -0.19558027,
  -0.45768735,
  -1.1988037,
  -1.4428236,
  -0.64759225,
  -0.6951825,
  -0.066758275,
  0.2753865,
  0.44198763,
  -0.5004722,
  0.49417102,
  0.40152887,
  0.19156331,
  -0.20195074,
  -0.78548855,
  -0.6010278,
  0.43873742,
  0.60128105,
  0.009750693,
  0.29560378,
  0.15369748,
  -0.11502031,
  0.19575457,
  -0.084062815,
  -0.57200676,
  0.52209103,
  -0.8718636,
  0.24354279,
  -0.23001705,
  -0.26813045,
  0.033662252,
  0.41768172,
  -1.1656911,
  0.16492313,
  -0.23274249,
  -0.09856258,
  -0.14791532,
  0.3483261,
  -0.72370994,
  -0.46186447,
  1.002478,
  0.06690536,
  -0.6356916,
  -0.8068857,
  0.39828378,
  -0.987602,
  -0.48057964,
  -0.91443294,
  -1.1440288,
  -0.09522212,
  -0.36398962,
  -0.14455658,
  -0.09450873,
  0.49147022,
  0.37112617,
  0.1446139,
  -0.6551288,
  -0.10683546,
  1.0844564,
  -0.26117444,
  -0.61763823,
  -1.0215288,
  -0.47271717,
  1.4268862,
  0.51654065,
  -0.01449356,
  0.051638275,
  0.20195235,
  0.33145776,
  -0.47064057,
  0.03058975,
  0.44798565,
  0.043512702,
  -0.5963866,
  0.0999626,
  -0.78542507,
  -0.07645189,
  -0.35353783,
  -0.3817216,
  -0.250955,
  -0.71230185,
  -0.7246541,
  -0.09184249,
  -0.15457578,
  -0.47047973,
  0.16415931,
  0.4615652,
  0.9969347,
  1.1440169,
  0.11543016,
  -0.40084806,
  0.8114938,
  0.053596105,
  0.072810955,
  -0.48538062,
  0.6282832,
  -0.5842784,
  -0.8101653,
  0.7857535,
  0.11797458,
  -0.1375664,
  -0.5850693,
  0.60943407,
  0.40796915,
  -0.6434871,
  0.3112606,
  0.2279547,
  -0.8009926,
  -0.79564685,
  0.37275973,
  -0.48306096,
  0.6792743,
  0.7707857,
  0.33682466,
  0.6252336,
  0.9911188,
  0.18310043,
  -0.5297898,
  1.1664978,
  0.79496926,
  0.28931668,
  1.2245488,
  -0.8181311,
  1.2116734,
  -0.15652189,
  -1.0386447,
  0.40046248,
  0.36049753,
  0.30607986,
  0.3364994,
  -0.28523347,
  0.35157678,
  0.8339798,
  0.89151037,
  -0.26233238,
  0.8933543,
  0.84345686,
  1.1897295,
  -0.48857647,
  0.79497045,
  0.10652572,
  -0.4135232,
  0.2182443,
  -0.35306656,
  0.42068833,
  -0.79997915,
  0.5809158,
  1.3345989,
  0.060842063,
  -0.2563161,
  0.03534407,
  0.5360344,
  1.917547,
  -0.36253482,
  -0.49243537,
  0.26489013,
  -0.104902044,
  -0.9972624,
  -0.091595255,
  0.46784645,
  -0.36084253,
  0.54299957,
  -0.3643171,
  -0.5715756,
  0.19559705,
  -0.5812822,
  -0.24326274,
  -0.6559266,
  -0.350655,
  0.44452924,
  -0.036831643,
  0.73498416,
  -0.25966042,
  -0.10833143,
  0.10960496,
  -0.08538726,
  -1.2817942,
  0.8961285,
  0.458948,
  -0.6589153,
  -0.44616297,
  -0.41166815,
  -0.04059479,
  1.2963586,
  -0.56560487,
  0.6094889,
  0.7305699,
  0.21042939,
  0.061067816,
  0.13612866,
  0.22969145,
  -0.17407487,
  -0.3607265,
  0.9569114,
  -1.2538174,
  -0.89215136,
  0.06465889,
  0.065436125,
  -0.633111,
  -0.50003904,
  1.043686,
  -1.4567229,
  -1.008188,
  1.406193,
  0.3361634,
  1.3332826,
  0.537974,
  0.06511741,
  0.48550525,
  0.20300093,
  -1.051667,
  0.38812464,
  -0.2860467,
  -0.022678724,
  0.28437257,
  0.71422035,
  0.82023406,
  0.47109297,
  0.35143635,
  -0.03959342,
  -0.5108114,
  -0.91239876,
  -0.7520437,
  0.6948058,
  -0.45881596,
  -0.6100966,
  0.46948856,
  1.1021845,
  -1.01778,
  -1.4687196,
  0.7579906,
  -0.07827543,
  0.57677615,
  -0.10651896,
  -0.11345732,
  -1.0104455,
  0.34969953,
  -0.45930463,
  -0.4082237,
  0.6603107,
  0.61721253,
  -0.11197656,
  0.37283185,
  -0.75463307,
  -0.5460384,
  0.3751441,
  0.2643444,
  -0.35107365,
  -0.4929604,
  -0.08177016
]
```

#### Error Responses

When using dynamic batching, errors are returned with HTTP response code 400 and content:

``` 
{
  "code": 400,
  "type": "TranslateException",
  "message": "Missing \"inputs\" in json."
}
```
