import argparse
from huggingface_hub import snapshot_download
from pathlib import Path


def llm_download(model_id, token, allow_patterns, download_dir):
    local_model_path = Path(download_dir)
    local_model_path.mkdir(exist_ok=True)
    ignore_patterns = [".git*"]
    snapshot_download(repo_id=model_id,
                      local_dir=local_model_path,
                      allow_patterns=allow_patterns,
                      ignore_patterns=ignore_patterns,
                      token=token)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build the download configs')
    parser.add_argument('model_id', help='model id from buggingface')
    parser.add_argument('--token',
                        required=False,
                        type=str,
                        help='The HuggingFace token have access to the model')
    parser.add_argument(
        '--allow-patterns',
        required=False,
        type=str,
        default='*.json,*.pt,*.safetensors,*.txt,*.model,*.tiktoken',
        help='The components to download')
    parser.add_argument('--download-dir',
                        required=False,
                        type=str,
                        default='./model',
                        help='directory to download model artifacts to')
    args = parser.parse_args()
    allow_patterns = args.allow_patterns.replace(" ", "").split(",")
    llm_download(args.model_id, args.token, allow_patterns, args.download_dir)
