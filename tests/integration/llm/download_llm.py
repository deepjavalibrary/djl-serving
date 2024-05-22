import argparse
from huggingface_hub import snapshot_download
from pathlib import Path


def llm_download(model_id, token, allow_patterns):
    local_model_path = Path("./model")
    local_model_path.mkdir(exist_ok=True)
    if not allow_patterns:
        # Only download pytorch checkpoint files
        allow_patterns = ["*.json", "*.pt", "*.safetensors", "*.txt", "*.model", "*.tiktoken"]

    snapshot_download(
        repo_id=model_id,
        local_dir=local_model_path,
        allow_patterns=allow_patterns,
        token=token
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the download configs')
    parser.add_argument('model_id', help='model id from buggingface')
    parser.add_argument('--token',
                        required=False,
                        type=str,
                        help='The HuggingFace token have access to the model')
    parser.add_argument('--allow-patterns',
                        required=False,
                        type=str,
                        help='The components to download')
    args = parser.parse_args()
    allow_patterns = args.allow_patterns.replace(" ", "").split(",")
    llm_download(args.model_id, args.token, allow_patterns)
