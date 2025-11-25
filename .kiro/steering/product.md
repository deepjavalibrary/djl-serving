# DJL Serving - Product Overview

High-performance universal model serving solution powered by Deep Java Library (DJL). Serves ML models through REST APIs with automatic scaling, dynamic batching, and multi-engine support.

## Architecture

**3-Layer Design:**
1. **Frontend** - Netty HTTP server (Inference + Management APIs)
2. **Workflows** - Multi-model execution pipelines
3. **WorkLoadManager (WLM)** - Worker thread pools with batching/routing

**Python Engine** - Runs Python-based models and custom handlers
**LMI** - Large Model Inference with vLLM, TensorRT-LLM, HuggingFace Accelerate

## Supported Models

PyTorch TorchScript, SKLearn models, ONNX, Python scripts, XGBoost, Sentencepiece, HuggingFace models

## Primary Use Cases

1. **LLM Serving** - Optimized backends (vLLM, TensorRT-LLM) with LoRA adapters
2. **Multi-Model Endpoints** - Version management, workflows
3. **Custom Handlers** - Python preprocessing/postprocessing
4. **Embeddings & Multimodal** - Text embeddings, vision-language models
5. **AWS Integration** - SageMaker deployment, Neo optimization (compilation, quantization, sharding)

## Key Features

- Auto-scaling worker threads based on load
- Dynamic batching for throughput optimization
- Multi-engine support (serve different frameworks simultaneously)
- Plugin architecture for extensibility
- OpenAPI-compatible REST endpoints
