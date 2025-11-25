# Project Structure

## Root Layout

```
djl-serving/
├── serving/          # Main HTTP server and core serving logic
├── wlm/              # WorkLoadManager - worker thread management
├── engines/python/   # Python engine for custom handlers and LLM backends
├── plugins/          # Extensible plugin system
├── benchmark/        # Performance benchmarking tools
├── prometheus/       # Prometheus metrics integration
├── awscurl/          # AWS request signing utility
├── tests/            # Integration and end-to-end tests
├── tools/            # Build and quality tools (checkstyle, PMD configs)
└── gradle/           # Gradle wrapper and version catalog
```

## Key Modules

### serving/ - HTTP Server & Core Logic
- `src/main/java/ai/djl/serving/` - Server, handlers, model management
- `src/main/conf/` - config.properties, log4j2.xml
- `docs/` - API documentation, configuration guides
- `docker/` - Dockerfiles, partition scripts

### engines/python/ - Python Engine & LLM Backends
- `setup/djl_python/` - Python package
  - `lmi_vllm/` - vLLM integration
  - `lmi_trtllm/` - TensorRT-LLM integration
  - `rolling_batch/` - Continuous batching
  - `chat_completions/` - OpenAI-compatible API
  - `adapter_manager_mixin.py` - LoRA adapter management
  - `service_loader.py` - Dynamic service loading

### wlm/ - WorkLoadManager
Worker pools, job queues, batching logic at `src/main/java/ai/djl/serving/wlm/`

### plugins/ - Extensibility
`cache/`, `kserve/`, `management-console/`, `secure-mode/`, `static-file-plugin/`

### tests/ - Integration Tests
- `integration/tests.py` - Main pytest suite
- `integration/llm/` - LLM test clients
- `integration/examples/` - Custom handler examples

## Configuration Conventions

### Java

- Package structure: `ai.djl.serving.*`
- Configuration: `config.properties` files
- Logging: Log4j2 XML configurations
- License header required on all Java files (Apache 2.0)

### Python

- Package: `djl_python`
- Entry point: `djl_python.test_model:run`
- Configuration: `serving.properties` in model directory
- Type hints encouraged for public APIs

### Docker

- Base images in `serving/docker/`
- Entrypoint: `dockerd-entrypoint.sh`
- Model location: `/opt/ml/model/`
- DJL home: `/opt/djl/`
- Python packages: `/opt/djl/python/djl_python/`
- Partition scripts: `/opt/djl/partition/` (copied from `serving/docker/partition/`)

## Important Paths

### In Container

- `/opt/djl/` - DJL Serving installation
- `/opt/ml/model/` - Model artifacts directory
- `/tmp/.djl.ai/` - DJL cache directory
- `/tmp/.cache/huggingface/` - HuggingFace cache

### In Repository

- `gradle/libs.versions.toml` - Centralized dependency versions
- `settings.gradle.kts` - Module definitions
- `buildSrc/` - Custom Gradle plugins
- `.github/workflows/` - CI/CD pipelines

## Naming Conventions

### Java

- Classes: PascalCase (e.g., `ModelServer`, `WorkerPoolConfig`)
- Methods: camelCase (e.g., `registerWorkflow`, `loadModels`)
- Constants: UPPER_SNAKE_CASE (e.g., `MODEL_SERVER_HOME`)

### Python

- Modules: snake_case (e.g., `service_loader.py`, `adapter_manager_mixin.py`)
- Classes: PascalCase (e.g., `VllmAsyncService`, `AdapterManagerMixin`)
- Functions: snake_case (e.g., `load_model`, `format_input`)
- Mixins: Suffix with `Mixin` (e.g., `AdapterFormatterMixin`)

### Configuration

- Environment variables: UPPER_SNAKE_CASE (e.g., `TENSOR_PARALLEL_DEGREE`, `CUDA_VISIBLE_DEVICES`)
- Properties: snake_case or dot notation (e.g., `option.tensor_parallel_degree`, `engine`)

## Testing Structure

- Java tests mirror source structure in `src/test/java/`
- Python tests in `djl_python/tests/` or alongside modules
- Integration tests use pytest with markers: `@pytest.mark.gpu`, `@pytest.mark.vllm`, `@pytest.mark.lora`
- Test models in `tests/integration/models/`
