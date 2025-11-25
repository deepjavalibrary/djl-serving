# Development Guidelines

## Repository Access

DJL, DJL-Serving, and LMI are open-source projects under the deepjavalibrary GitHub organization.

### Getting Started

1. Complete Open Source Training
2. Link GitHub account with AWS/Amazon
3. Join the deepjavalibrary GitHub organization
4. Request access to djl-admin or djl-committer groups

### Key Repositories

- https://github.com/deepjavalibrary/djl
- https://github.com/deepjavalibrary/djl-serving
- https://github.com/deepjavalibrary/djl-demo

## Development Workflow

### Setup
```bash
# Fork repos, then clone and track upstream
git clone git@github.com:<username>/djl-serving.git
cd djl-serving
git remote add upstream https://github.com/deepjavalibrary/djl-serving

# Sync with upstream
git fetch upstream && git rebase upstream/master && git push
```

### Making Changes
```bash
git checkout -b my-feature-branch
# Make changes
git add . && git commit -m "Description"
git push -u origin my-feature-branch
# Create PR from fork to upstream/master via GitHub UI
```

## Building LMI Containers

### Container Types

**DLC and DockerHub:**
- LMI-vLLM
- LMI-TensorRT-LLM
- LMI-Neuron

**DockerHub Only:**
- CPU-Full (PyTorch/OnnxRuntime/MxNet/TensorFlow)
- CPU (no engines bundled)
- PyTorch-GPU
- Aarch64 (Graviton support)

### Build Process

```bash
# Prepare build
cd djl-serving
rm -rf serving/docker/distributions
./gradlew clean && ./gradlew --refresh-dependencies :serving:dockerDeb -Psnapshot

# Get versions
cd serving/docker
export DJL_VERSION=$(awk -F '=' '/djl / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)
export SERVING_VERSION=$(awk -F '=' '/serving / {gsub(/ ?"/, "", $2); print $2}' ../../gradle/libs.versions.toml)

# Build specific container
docker compose build --build-arg djl_version=${DJL_VERSION} --build-arg djl_serving_version=${SERVING_VERSION} lmi
docker compose build --build-arg djl_version=${DJL_VERSION} --build-arg djl_serving_version=${SERVING_VERSION} tensorrt-llm
docker compose build --build-arg djl_version=${DJL_VERSION} --build-arg djl_serving_version=${SERVING_VERSION} pytorch-inf2
```

See `serving/docker/docker-compose.yml` for all available targets.

## Testing

### Local Integration Tests

```bash
cd tests/integration
OVERRIDE_TEST_CONTAINER=<image_name> python -m pytest tests.py::<TestClass>::<test_name>

# Example
OVERRIDE_TEST_CONTAINER=deepjavalibrary/djl-serving:lmi python -m pytest tests.py::TestVllm1_g6::test_gemma_2b
```

Full test suite: `tests/integration/tests.py`

## Key Development Areas (Priority Order)

### DJL-Serving
1. **Python Engine** - `engines/python/setup/djl_python/` (vLLM, TensorRT-LLM, rolling batch, chat completions)
2. **Python Engine Java** - `engines/python/src/main/java/ai/djl/python/engine/`
3. **WLM** - `wlm/` (backend ML/DL engine integration)
4. **Serving** - `serving/` (frontend web server)

### DJL (Less Frequent)
PyTorch, HuggingFace Tokenizer, OnnxRuntime, Rust/Candle engines

## CI/CD Workflows

### DJL Repository
- `continuous.yml` - PR checks
- `native_jni_s3_pytorch.yml` - Publish native code to S3
- `nightly_publish.yml` - SNAPSHOT to Maven
- `serving-publish.yml` - DJL-Serving SNAPSHOT to S3

### DJL-Serving Repository
- `nightly.yml` - Build containers → Run tests → Publish to staging
- `docker-nightly-publish.yml` - Build/publish to dev repo (ad-hoc)
- `integration.yml` - Run all tests with custom image (ad-hoc)
- `docker_publish.yml` - Sync dev to staging
- `integration_execute.yml` - Single test on specific instance

## Versioning
- **DJL** → Maven (stable + SNAPSHOT)
- **DJL-Serving** → S3 (stable + SNAPSHOT)
- **Source** → `gradle/libs.versions.toml`
- **Nightly** → SNAPSHOT, **Release** → Stable
