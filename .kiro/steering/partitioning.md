# Model Partitioning and Optimization

The partition system (`serving/docker/partition/`) provides tools for model preparation, including tensor parallelism sharding, quantization, and multi-node setup.

## Core Scripts

### partition.py - Main Entry Point
Handles S3 download, requirements install, partitioning, quantization (AWQ/FP8), S3 upload.

**Features:** HF downloads, `OPTION_*` env vars, MPI mode, auto-cleanup

```bash
python partition.py \
  --model-id <hf_model_id_or_s3_uri> \
  --tensor-parallel-degree 4 \
  --quantization awq \
  --save-mp-checkpoint-path /tmp/output
```

### run_partition.py - Custom Handlers
Invokes user-provided partition handlers via `partition_handler` property.

### run_multi_node_setup.py - Cluster Coordination
Multi-node setup: queries leader for model info, downloads to workers, exchanges SSH keys, reports readiness.

**Env Vars:** `DJL_LEADER_ADDR`, `LWS_LEADER_ADDR`, `DJL_CACHE_DIR`

### trt_llm_partition.py - TensorRT-LLM Compilation
Builds TensorRT engines with BuildConfig (batch/seq limits), QuantConfig (AWQ/FP8/SmoothQuant), CalibConfig (calibration data).

### SageMaker Neo Integration

Partition scripts power **SageMaker Neo's CreateOptimizationJob API** - managed service for compilation, quantization, and sharding.

**API:** https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_CreateOptimizationJob.html

**Optimization Types:**
1. Compilation (TensorRT-LLM engines)
2. Quantization (AWQ, FP8)
3. Sharding (Fast Model Loader TP)

**Neo Environment Variables:**
- `SM_NEO_INPUT_MODEL_DIR`, `SM_NEO_OUTPUT_MODEL_DIR`
- `SM_NEO_COMPILATION_PARAMS` (JSON config)
- `SERVING_FEATURES` (vllm, trtllm)

**Neo Scripts:**
- `sm_neo_dispatcher.py` - Routes jobs: vllm→Quantize/Shard, trtllm→Compile
- `sm_neo_trt_llm_partition.py` - TensorRT-LLM compilation
- `sm_neo_quantize.py` - Quantization workflows
- `sm_neo_utils.py` - Env var helpers

**Workflow:**
CreateOptimizationJob(source S3, config, output S3, container) → Neo launches container → Dispatcher routes → Handler optimizes → Artifacts to output S3 → Deploy to SageMaker

## Quantization

### AWQ (4-bit, AutoAWQ library)
```properties
option.quantize=awq
option.awq_zero_point=true
option.awq_block_size=128
option.awq_weight_bit_width=4
option.awq_mm_version=GEMM
option.awq_ignore_layers=lm_head
```

### FP8 (llm-compressor, CNN/DailyMail calibration)
```properties
option.quantize=fp8
option.fp8_scheme=FP8_DYNAMIC
option.fp8_ignore=lm_head
option.calib_size=512
option.max_model_len=2048
```

## Multi-Node

### MPI Mode (engine=MPI or TP > 1)
```bash
mpirun -N <tp_degree> --allow-run-as-root \
  --mca btl_vader_single_copy_mechanism none \
  -x FI_PROVIDER=efa -x RDMAV_FORK_SAFE=1 \
  python run_partition.py --properties '{...}'
```

### Cluster Setup (LeaderWorkerSet/K8s)
1. Leader generates SSH keys
2. Workers query `/cluster/models` for model info
3. Workers download model, exchange SSH keys via `/cluster/sshpublickey`
4. Workers report to `/cluster/status?message=OK`
5. Leader loads model

## Configuration

### properties_manager.py
Loads `serving.properties`, merges `OPTION_*` env vars, validates, generates output.

**Key Properties:**
- `option.model_id` - HF model ID or S3 URI
- `option.tensor_parallel_degree`, `option.pipeline_parallel_degree`
- `option.save_mp_checkpoint_path` - Output dir
- `option.quantize` - awq, fp8, static_int8
- `engine` - Python, MPI

### utils.py Helpers
`get_partition_cmd()`, `extract_python_jar()`, `load_properties()`, `update_kwargs_with_env_vars()`, `remove_option_from_properties()`, `load_hf_config_and_tokenizer()`

## Container Integration
Scripts at `/opt/djl/partition/` invoked via:
1. Neo compilation (`sm_neo_dispatcher.py`)
2. Container startup (on-the-fly partitioning)
3. Management API (dynamic registration)

## Common Workflows

```bash
# Tensor Parallelism
python partition.py --model-id meta-llama/Llama-2-70b-hf \
  --tensor-parallel-degree 8 --save-mp-checkpoint-path /tmp/output

# AWQ Quantization
python partition.py --model-id meta-llama/Llama-2-7b-hf \
  --quantization awq --save-mp-checkpoint-path /tmp/output

# TensorRT-LLM Engine
python trt_llm_partition.py --properties_dir /opt/ml/model \
  --trt_llm_model_repo /tmp/engine --model_path /tmp/model \
  --tensor_parallel_degree 4 --pipeline_parallel_degree 1
```

## Error Handling
Non-zero exit on failure, real-time stdout/stderr, cleanup on success, S3 upload only after success.
