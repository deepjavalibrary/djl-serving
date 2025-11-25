# Technology Stack

## Build System

**Gradle** (Kotlin DSL) - Multi-module project with custom plugins

## Languages

- **Java 17+** - Core serving infrastructure, WLM, plugins
- **Python 3.9+** - Model handlers, inference engines (vLLM, TensorRT-LLM)
- **Shell** - Docker entrypoints, deployment scripts

## Core Java Stack

- **Netty** - Async HTTP server and networking
- **DJL (Deep Java Library)** - ML framework abstraction layer
- **SLF4J + Log4j2** - Logging
- **Apache Commons CLI** - Command-line parsing
- **TestNG** - Java testing
- **Mockito** - Mocking framework

## Python Stack

- **PyTorch** - Primary deep learning framework
- **vLLM** - High-performance LLM inference (v0.10.2)
- **TensorRT-LLM** - NVIDIA optimized LLM inference
- **Transformers** - HuggingFace model support
- **PEFT** - Parameter-efficient fine-tuning (LoRA adapters)
- **pytest** - Python testing
- **setuptools** - Python packaging

## Container & Deployment

- **Docker** - Multi-stage builds for CPU, GPU, LMI variants
- **CUDA 12.8** - GPU support
- **Ubuntu 24.04** - Base container OS
- **AWS SageMaker** - Primary deployment target

## Common Commands

### Build
```bash
./gradlew build                    # Full build
./gradlew :serving:build           # Specific module
./gradlew build -x test            # Skip tests
./gradlew :serving:distTar         # Create distribution
```

### Test
```bash
# Java tests
./gradlew test
./gradlew :serving:test

# Python tests (from engines/python/setup/)
pytest -v
pytest -k test_name

# Integration tests (from tests/integration/)
pytest tests.py::TestVllm1_g6::test_gemma_2b
OVERRIDE_TEST_CONTAINER=<image> pytest tests.py::<TestClass>::<test>
```

### Python Development
```bash
cd engines/python/setup
pip install -e ".[test]"           # Editable install with test deps
pytest djl_python/tests/           # Run tests
```

### Docker
```bash
# Build container (see development.md for full process)
cd serving/docker
docker compose build --build-arg djl_version=${DJL_VERSION} lmi

# Run container
docker run -it --rm --gpus all -p 8080:8080 \
  -v /path/to/model:/opt/ml/model \
  deepjavalibrary/djl-serving:0.34.0-lmi
```

### Local Server
```bash
./gradlew :serving:run                      # Default
djl-serving -m test=file:/path/to/model     # With model
djl-serving -f config.properties            # With config
```

## Code Quality Tools

- **Checkstyle** - Java code style enforcement
- **SpotBugs** - Static analysis
- **PMD** - Code quality checks
- **YAPF** - Python code formatting

## Version Management

Version defined in `gradle/libs.versions.toml` and synchronized across Java and Python modules.
