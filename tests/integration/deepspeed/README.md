# Add test models (DJL internal)

For DeepSpeed container testing, we will download model from DJL's S3 bucket.
To check in new models, follow the steps below:

```python
from huggingface_hub import snapshot_download
from pathlib import Path

# - This will download the model into the ./model directory where ever the jupyter file is running
local_model_path = Path("./model")
local_model_path.mkdir(exist_ok=True)
model_name = "facebook/opt-30b"
# Only download pytorch checkpoint files
allow_patterns = ["*.json", "*.pt", "*.bin", "*.txt"]

# - Leverage the snapshot library to donload the model since the model is stored in repository using LFS
snapshot_download(
    repo_id=model_name,
    cache_dir=local_model_path,
    allow_patterns=allow_patterns,
)
```

After that, please store the model to:

```
aws s3 sync model/ s3://djl-llm/opt-30b/
```
