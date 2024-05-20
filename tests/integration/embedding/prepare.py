import argparse
import os
import shutil

onnx_list = {
    "bge-base-en-v1.5": {
        "option.model_id": "BAAI/bge-base-en-v1.5",
        "batch_size": 32,
    },
    "bge-reranker": {
        "option.model_id": "BAAI/bge-reranker-base",
        "batch_size": 32,
    }
}


def write_model_artifacts(properties, requirements=None):
    model_path = "models/test"
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "serving.properties"), "w") as f:
        for key, value in properties.items():
            f.write(f"{key}={value}\n")
    if requirements:
        with open(os.path.join(model_path, "requirements.txt"), "w") as f:
            f.write('\n'.join(requirements) + '\n')


def build_onnx_model(model):
    if model not in onnx_list:
        raise ValueError(
            f"{model} is not one of the supporting handler {list(onnx_list.keys())}"
        )
    options = onnx_list[model]
    options["engine"] = "OnnxRuntime"
    options["option.mapLocation"] = "true"
    options["padding"] = "true"
    options["pooling"] = "cls"
    options["includeTokenTypes"] = "false"
    options["maxLength"] = 512
    options[
        "translatorFactory"] = "ai.djl.huggingface.translator.TextEmbeddingTranslatorFactory"
    options["normalize"] = "false"

    write_model_artifacts(options)


supported_engine = {
    'onnxruntime': build_onnx_model,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Build the Text Embedding configs')
    parser.add_argument('engine',
                        type=str,
                        choices=['onnxruntime'],
                        help='The engine used for inference')
    parser.add_argument('model',
                        type=str,
                        help='model that works with certain engine')
    parser.add_argument('--dtype',
                        required=False,
                        type=str,
                        help='The model data type')
    global args
    args = parser.parse_args()

    if args.engine not in supported_engine:
        raise ValueError(
            f"{args.engine} is not one of the supporting engine {list(supported_engine.keys())}"
        )
    supported_engine[args.engine](args.model)
