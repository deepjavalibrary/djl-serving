# DJL Serving - Secure Mode Plugin

This plugin implements SageMaker Secure Mode for the model server, performing security checks for potentially unsafe files and options. It is configured by the SageMaker platform.

## Overview

Secure Mode validates both **models** and **adapters** (LoRA) to prevent:
- Custom code execution via `model.py`
- Arbitrary package installation via `requirements.txt`
- Pickle deserialization exploits via `.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl` files
- Jinja template injection via `chat_template` in tokenizer configs
- Untrusted remote code execution

## Environment Variables

### Platform-Controlled (SageMaker Only)

- `SAGEMAKER_SECURE_MODE` - Enable/disable secure mode (`true`/`false`)
- `SAGEMAKER_SECURITY_CONTROLS` - Comma-separated list of controls to enforce
- `SAGEMAKER_UNTRUSTED_CHANNELS` - Comma-separated paths to scan for forbidden files

### User-Configurable (Validated by Plugin)

All model options are validated against an allowlist. Only ~100 explicitly allowed properties can be set.

## Security Controls

### 1. Custom Inference Scripts (DISALLOW_CUSTOM_INFERENCE_SCRIPTS)

Blocks custom Python entrypoints:
- `model.py` in model/adapter directory
- `option.entryPoint` not starting with `djl_python.`
- Only allows built-in DJL Python handlers

### 2. Trust Remote Code (DISALLOW_TRUST_REMOTE_CODE)

Blocks `option.trust_remote_code=true`:
- Prevents execution of arbitrary code from HuggingFace Hub
- Ensures only vetted model architectures are used

### 3. Pickle Files (DISALLOW_PICKLE_FILES)

Blocks pickle-based model files:
- `.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl` extensions
- Scans recursively in untrusted directories
- Only `.safetensors` format allowed

### 4. Chat Templates (DISALLOW_CHAT_TEMPLATE)

Blocks Jinja templates in `tokenizer_config.json`:
- Prevents template injection attacks
- Scans `chat_template` field in tokenizer configs

### 5. Requirements.txt (DISALLOW_REQUIREMENTS_TXT)

Blocks `requirements.txt` in model/adapter directory:
- Prevents installation of arbitrary Python packages
- Ensures only pre-installed dependencies are used

### 6. Python Executable Restriction

Only allows specific Python executables:
- `/opt/djl/vllm_venv/bin/python`
- `/usr/bin/python3`

Custom Python paths are blocked.

## Adapter Security

Adapters (LoRA) are validated using the same security controls as models. Validation happens at adapter creation time, before the adapter is registered.

### Validation Points

**Static Adapters (Workflow Config):**
```
AdapterWorkflowFunction.prepare()
  ↓
Adapter.newInstance() → SecureModeAdapterValidator.validateAdapterPath()
  ↓
[Pass] → Adapter registered
[Fail] → IllegalConfigurationException → Workflow fails
```

**Dynamic Adapters (Management API):**
```
POST /models/{model}/adapters
  ↓
AdapterManagementRequestHandler.handleRegisterAdapter()
  ↓
Adapter.newInstance() → SecureModeAdapterValidator.validateAdapterPath()
  ↓
[Pass] → HTTP 200 → Adapter registered
[Fail] → HTTP 400 → Adapter NOT registered
```

### Adapter Validation Rules

When `SAGEMAKER_SECURE_MODE=true`, adapters are checked for:

1. **DISALLOW_CUSTOM_INFERENCE_SCRIPTS** → Blocks `model.py` in adapter directory
2. **DISALLOW_REQUIREMENTS_TXT** → Blocks `requirements.txt` in adapter directory
3. **DISALLOW_PICKLE_FILES** → Blocks `.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl` files (recursive scan)

Failed adapters:
- Are NOT created
- Are NOT registered in ModelInfo
- Do NOT appear in LIST API (`GET /models/{model}/adapters`)
- Return clear error messages

## Engine Scope

Validation only applies to:
- **Python** engine
- **MPI** engine

Other engines (PyTorch, ONNX, etc.) skip security checks.

## Usage Examples

### Valid Configuration

```properties
# serving.properties
engine=Python
option.model_id=meta-llama/Llama-2-7b-hf
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=vllm
option.max_rolling_batch_size=32
option.enable_lora=true
```

### Valid Adapter

```
adapters/
  my-adapter/
    adapter_config.json
    adapter_model.safetensors  ✅ Allowed
```

### Invalid Configurations

**Custom entrypoint:**
```properties
option.entryPoint=model.py  # ❌ Blocked by DISALLOW_CUSTOM_INFERENCE_SCRIPTS
```

**Trust remote code:**
```properties
option.trust_remote_code=true  # ❌ Blocked by DISALLOW_TRUST_REMOTE_CODE
```

**Non-allowlisted property:**
```properties
option.custom_setting=value  # ❌ Not in allowlist
```

**Pickle file in untrusted directory:**
```
/opt/ml/model/pytorch_model.bin  # ❌ Blocked by DISALLOW_PICKLE_FILES
```

**Invalid adapter:**
```
adapters/
  bad-adapter/
    model.py  # ❌ Blocked by DISALLOW_CUSTOM_INFERENCE_SCRIPTS
    requirements.txt  # ❌ Blocked by DISALLOW_REQUIREMENTS_TXT
    adapter.bin  # ❌ Blocked by DISALLOW_PICKLE_FILES
```

## Error Handling

When validation fails, the plugin throws `IllegalConfigurationException` with a descriptive message:

```
IllegalConfigurationException: Setting TRUST_REMOTE_CODE to True is prohibited in Secure Mode.
IllegalConfigurationException: Pickle-based file adapter.bin found in adapter at /path/to/adapter, but only the Safetensors format is permitted in Secure Mode.
IllegalConfigurationException: Custom model.py found in adapter at /path/to/adapter, but custom inference scripts are prohibited in Secure Mode.
IllegalConfigurationException: Property option.custom_option is prohibited from being set in Secure Mode.
```

The model/adapter will not load, and the error is returned to the caller.

## Testing

### Run Unit Tests

```bash
# All secure mode tests (38 tests)
./gradlew :plugins:secure-mode:test

# Adapter security tests only (17 tests)
./gradlew :plugins:secure-mode:test --tests SecureModeAdapterTest

# Or use the helper script
./plugins/secure-mode/run-adapter-tests.sh
```

### Test Coverage

**Model Security Tests (21 tests):**
- Property allowlist enforcement
- Each security control independently
- Environment variable vs properties file configuration
- Valid configurations that should pass
- Engine-specific validation (Python/MPI only)

**Adapter Security Tests (17 tests):**
- Adapter with `model.py` → blocked
- Adapter with `requirements.txt` → blocked
- Adapter with pickle files (`.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl`) → blocked
- Adapter with `.safetensors` → allowed
- Adapter with clean files → allowed
- Secure mode disabled → all allowed
- No security controls set → all allowed
- Multiple violations → fails on first
- Recursive subdirectory scanning
- Invalid adapter paths
- Static adapter loading
- Dynamic adapter registration

## Implementation Details

### Core Classes

- `SecureModePlugin` - Plugin entry point, registers listener
- `SecureModeModelServerListener` - Hooks into model configuration events
- `SecureModeUtils` - Shared validation logic for models and adapters
- `SecureModeAdapterValidator` - Adapter-specific validation
- `SecureModeAllowList` - Defines allowlisted properties and executables

### Validation Flow

**Model Validation:**
1. Check if secure mode is enabled (`SAGEMAKER_SECURE_MODE=true`)
2. Verify security controls are configured
3. Skip validation for non-Python/MPI engines
4. Validate properties against allowlist
5. Check options based on enabled controls
6. Scan untrusted directories for forbidden files
7. Throw exception on any violation

**Adapter Validation:**
1. Check if secure mode is enabled
2. Verify security controls are configured
3. Validate adapter directory exists
4. Check for `model.py` (if DISALLOW_CUSTOM_INFERENCE_SCRIPTS)
5. Check for `requirements.txt` (if DISALLOW_REQUIREMENTS_TXT)
6. Scan recursively for pickle files (if DISALLOW_PICKLE_FILES)
7. Throw exception on any violation

### Code Organization

The implementation follows DRY principles with shared validation methods:

- `SecureModeUtils.checkModelPy()` - Used by both model and adapter validation
- `SecureModeUtils.checkRequirementsTxt()` - Used by both model and adapter validation
- `SecureModeUtils.scanPickleFiles()` - Used by both model and adapter validation

This ensures consistent behavior and error messages across models and adapters.

## SageMaker Integration

This plugin is automatically enabled by SageMaker when deploying models with security restrictions. Users cannot manually enable or configure it - all settings are controlled by the SageMaker platform.

**Typical SageMaker workflow:**
1. User creates SageMaker endpoint with security requirements
2. SageMaker sets environment variables in container
3. DJL Serving starts with Secure Mode plugin
4. Plugin validates model configuration on load
5. Plugin validates adapters on registration (static or dynamic)
6. Model serves requests (if validation passes) or fails fast (if validation fails)

## Limitations

- Only validates Python and MPI engines
- Cannot be enabled outside of SageMaker platform
- Allowlist is fixed at compile time (requires code change to modify)
- File scanning is recursive and may impact startup time for large model directories
- Adapter validation happens at creation time, not at inference time

## Development

### Adding New Allowlisted Properties

1. Add to `SecureModeAllowList.PROPERTIES_ALLOWLIST`
2. Add test case in `SecureModePluginTest`
3. Update documentation

### Adding New Security Controls

1. Define constant in `SecureModeUtils`
2. Add validation logic in `checkOptions()` or `scanForbiddenFiles()`
3. Add validation in `SecureModeAdapterValidator` if applicable
4. Add comprehensive test cases
5. Update documentation

## References

- Plugin system: `serving/docs/plugin_management.md`
- Model configuration: `serving/docs/configuration.md`
- Adapter management: `serving/docs/lmi/user_guides/lora_adapter_guide.md`
- SageMaker integration: AWS SageMaker documentation
