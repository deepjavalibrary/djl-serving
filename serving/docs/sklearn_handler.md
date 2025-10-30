# Scikit-learn Handler

The scikit-learn handler enables serving scikit-learn models with DJL Serving's Python engine. It supports multiple model formats and provides flexible customization options.

## Supported Model Formats

### Secure Formats (Require option.skops_trusted_types='list,of,types,separated,by,commas')
- **skops**: `.skops` files

### Insecure Formats (Require option.trust_insecure_model_files=true)
- **joblib**: `.joblib`, `.jl` files
- **pickle**: `.pkl`, `.pickle` files
- **cloudpickle**: `.pkl`, `.pickle`, `.cloudpkl` files

## Configuration

The scikit-learn handler accepts configurations in two formats:

* `serving.properties` Configuration File (per model configurations)
* Environment Variables (global configurations)

For most use-cases, using environment variables is sufficient.
Configurations specified in the `serving.properties` files will override configurations specified in environment variables.

### serving.properties Configuration

Create a `serving.properties` file:

```properties
engine=Python
option.entryPoint=djl_python.sklearn_handler
option.model_format=skops
option.skops_trusted_types=sklearn.ensemble._forest.RandomForestClassifier,numpy.ndarray
```

For insecure formats, add:
```properties
option.trust_insecure_model_files=true
```

### Environment Variable Configuration

Alternatively, configure via environment variables:

```python
env = {
    'OPTION_ENGINE': 'Python',
    'OPTION_ENTRY_POINT': 'djl_python.sklearn_handler',
    'OPTION_MODEL_FORMAT': 'skops',
    'OPTION_SKOPS_TRUSTED_TYPES': 'sklearn.ensemble._forest.RandomForestClassifier,numpy.ndarray',
    'OPTION_TRUST_INSECURE_MODEL_FILES': 'false'
}
```

Configuration keys that start with `option.` can be specified as environment variables using the `OPTION_` prefix.
The configuration `option.<property>` is translated to environment variable `OPTION_<PROPERTY>`.

### Model Directory Structure

```
model/
├── serving.properties   # Optional: If absent from model directory, must set with ENV variables
├── model.skops         # Your scikit-learn model file
└── model.py           # Optional: Custom handlers
```

### Default Input/Output

The scikit-learn handler supports both JSON and CSV input/output formats.

**JSON Input Format:**
```json
{
  "inputs": [[1.0, 2.0, 3.0, 4.0, 5.0]]
}
```

Also accepts direct arrays format:
```json
[
  [1.0, 2.0, 3.0, 4.0, 5.0],
  [2.0, 3.0, 4.0, 5.0, 6.0]
]
```

**JSON Output Format:**
```json
{
  "predictions": [1]
}
```

**CSV Input Format:**
```
1.0,2.0,3.0,4.0,5.0
2.0,3.0,4.0,5.0,6.0
```

**CSV Output Format:**
```
1
0
```
Each prediction is returned on a separate line.

## Custom Formatters/Handlers

There are two ways to customize the scikit-learn handler behavior:

### Method 1: DJL Decorators

Create a `model.py` file using DJL decorators:

```python
from djl_python.input_parser import input_formatter, prediction_handler, init_handler
from djl_python.output_formatter import output_formatter
from djl_python import Input
import numpy as np
import joblib
import os

@init_handler
def custom_init(model_dir, **kwargs):
    """Custom model initialization"""
    model_path = os.path.join(model_dir, "model.skops")
    import skops.io as sio
    trusted_types = ['sklearn.ensemble._forest.RandomForestClassifier', 'numpy.ndarray']
    model = sio.load(model_path, trusted=trusted_types)
    return model

@input_formatter
def custom_input(inputs: Input, **kwargs):
    """Custom input processing - returns numpy array for default predict"""
    data = inputs.get_as_json()
    features = data.get("features", data.get("inputs"))
    X = np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

@prediction_handler
def custom_predict(X, model, **kwargs):
    """Custom prediction logic"""
    predictions = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        return {"predictions": predictions, "probabilities": probabilities}
    return predictions

@output_formatter
def custom_output(predictions):
    """Custom output formatting"""
    if isinstance(predictions, dict):
        return predictions
    return {"predictions": predictions.tolist()}
```

### Method 2: SageMaker-Style Functions

Alternatively, use SageMaker-compatible function signatures:

```python
import numpy as np
import joblib
import json
import os

def model_fn(model_dir):
    """Load model - model_dir is the directory name"""
    model_path = os.path.join(model_dir, "model.skops")
    import skops.io as sio
    trusted_types = ['sklearn.ensemble._forest.RandomForestClassifier', 'numpy.ndarray']
    model = sio.load(model_path, trusted=trusted_types)
    return model

def input_fn(request_body, request_content_type):
    """Parse input - request_body is byte buffer (request_content_type can also be named content_type)"""
    if request_content_type == 'application/json':
        data = json.loads(request_body.decode('utf-8'))
        features = data.get("features", data.get("inputs"))
        return np.array(features)  # Return numpy for default predict
    elif request_content_type == 'text/csv':
        import io
        data = np.loadtxt(io.StringIO(request_body.decode('utf-8')), delimiter=',')
        return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_object, model):
    """Run prediction"""
    return model.predict(input_object)

def output_fn(prediction, response_content_type):
    """Format output - returns byte array (response_content_type can also be named accept)"""
    if response_content_type == 'application/json':
        result = {"predictions": prediction.tolist()}
        return json.dumps(result).encode('utf-8')
    elif response_content_type == 'text/csv':
        return '\n'.join(map(str, prediction)).encode('utf-8')
    else:
        raise ValueError(f"Unsupported accept type: {response_content_type}")
```

### Important Notes

- **DJL decorators take precedence** over SageMaker functions - cannot mix both approaches
- **Omitted functions use defaults**: If any decorator/function is omitted, the handler uses default logic
- **Default predict expects numpy**: When using custom input formatters, return numpy arrays for default prediction
- **Default output processes numpy**: Default output formatting expects numpy arrays from prediction

### SageMaker Model Example
```python
from sagemaker import Model

# Environment variables for scikit-learn handler
env = {
    'SAGEMAKER_MODEL_SERVER_VMARGS': '-Xmx2g -Xms2g',
    'SAGEMAKER_STARTUP_TIMEOUT': '600',
    'SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS': '240',
    'SAGEMAKER_MAX_PAYLOAD_IN_MB': '10',
    'SAGEMAKER_NUM_MODEL_WORKERS': '1'
}

# Create SageMaker model
model = Model(
    image_uri='763104351884.dkr.ecr.us-west-2.amazonaws.com/djl-inference:0.35.0-cpu-full',
    model_data='s3://your-bucket/sklearn-model.tar.gz',
    role=role,
    env=env
)

# Deploy endpoint
predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.xlarge'
)
```

### Available Environment Variables

#### Handler Configuration
| Variable | Description | Default | Example |
|----------|-------------|---------|----------|
| `OPTION_ENTRY_POINT` | Handler entry point | - | `djl_python.sklearn_handler` |
| `OPTION_MODEL_FORMAT` | Model file format | - | `skops`, `joblib`, `pickle`, `cloudpickle` |
| `OPTION_SKOPS_TRUSTED_TYPES` | Trusted types for skops | - | `sklearn.ensemble._forest.RandomForestClassifier,numpy.ndarray` |
| `OPTION_TRUST_INSECURE_MODEL_FILES` | Allow insecure formats | `false` | `true` |

#### SageMaker-Specific Variables (Backwards Compatible)
These variables work with any DJL deployment - not just SageMaker endpoints.
| Variable | Description | Example |
|----------|-------------|----------|
| `SAGEMAKER_MAX_REQUEST_SIZE` | Max request size (bytes) | `10485760` |
| `SAGEMAKER_NUM_MODEL_WORKERS` | Number of model workers | `2` |
| `SAGEMAKER_DEFAULT_INVOCATIONS_ACCEPT` | Default accept header | `application/json` |
| `SAGEMAKER_MODEL_SERVER_TIMEOUT_SECONDS` | Model server prediction timeout | `240` |
| `SAGEMAKER_MODEL_SERVER_VMARGS` | JVM arguments | `-Xmx4g -Xms4g` |
| `SAGEMAKER_STARTUP_TIMEOUT` | Startup timeout (seconds) | `600` |
| `SAGEMAKER_MAX_PAYLOAD_IN_MB` | Max payload size (MB) | `10` |

## Troubleshooting

### Common Issues

1. **Model format not recognized**: Ensure file extension matches format
2. **Security error**: Set appropriate trusted types for skops or enable trust_insecure_model_files
3. **Input shape mismatch**: Verify input dimensions match training data or if using custom formatters, check formatter logic
4. **Import errors**: Ensure all required scikit-learn modules are available
