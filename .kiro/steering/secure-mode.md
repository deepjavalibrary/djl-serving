# Secure Mode Plugin

## Overview

The Secure Mode plugin (`plugins/secure-mode/`) implements security validation for SageMaker deployments. It restricts model configurations to a safe subset of options and validates files before models load.

## When to Work on This

- Adding new security controls for SageMaker requirements
- Expanding the properties allowlist for new features
- Fixing security vulnerabilities in model loading
- Updating file format validation rules
- SageMaker platform integration changes

## Architecture

**Event-Driven Design:**
- Plugin registers `SecureModeModelServerListener` on startup
- Listener intercepts `onModelConfigured()` events
- Validation runs before model loads
- Throws `IllegalConfigurationException` on any violation

**Key Components:**
- `SecureModePlugin` - Entry point, registers listener when `SAGEMAKER_SECURE_MODE=true`
- `SecureModeModelServerListener` - Event listener that triggers validation
- `SecureModeUtils` - Core validation logic (properties, options, file scanning)
- `SecureModeAllowList` - Defines allowlisted properties and Python executables

## Security Controls

### Control Types

1. **DISALLOW_REQUIREMENTS_TXT** - Blocks `requirements.txt` in model directory
2. **DISALLOW_PICKLE_FILES** - Only allows Safetensors (blocks .bin, .pt, .pth, .ckpt, .pkl)
3. **DISALLOW_TRUST_REMOTE_CODE** - Prevents `trust_remote_code=true`
4. **DISALLOW_CUSTOM_INFERENCE_SCRIPTS** - Only allows `djl_python.*` entrypoints
5. **DISALLOW_CHAT_TEMPLATE** - Blocks Jinja templates in `tokenizer_config.json`

### Validation Layers

**Layer 1: Properties Allowlist**
- Validates all properties against `PROPERTIES_ALLOWLIST`
- ~100 properties explicitly allowed
- Any other property throws exception

**Layer 2: Options Check**
- Validates specific options based on enabled controls
- Checks both properties file and environment variables
- Validates Python executable path

**Layer 3: File Scanning**
- Recursively scans `SAGEMAKER_UNTRUSTED_CHANNELS` directories
- Checks file extensions and content based on controls
- Parses JSON files for forbidden fields

## Environment Variables

**Platform-Controlled (SageMaker only):**
- `SAGEMAKER_SECURE_MODE` - Enable/disable (true/false)
- `SAGEMAKER_SECURITY_CONTROLS` - Comma-separated control list
- `SAGEMAKER_UNTRUSTED_CHANNELS` - Comma-separated paths to scan

**User-Configurable (validated by plugin):**
- `OPTION_*` - Model options (validated against allowlist)
- `DJL_ENTRY_POINT` - Entrypoint (must be `djl_python.*`)

## Development Guidelines

### Adding New Allowlisted Properties

1. Add to `SecureModeAllowList.PROPERTIES_ALLOWLIST`
2. Document in README
3. Add test case in `SecureModePluginTest`

```java
public static final Set<String> PROPERTIES_ALLOWLIST = Set.of(
    // ... existing properties
    "option.new_property"  // Add here
);
```

### Adding New Security Controls

1. Define constant in `SecureModeUtils`:
```java
static final String NEW_CONTROL = "DISALLOW_NEW_FEATURE";
```

2. Add validation logic in `checkOptions()` or `scanForbiddenFiles()`
3. Update README with control description
4. Add comprehensive test cases

### Testing Approach

**Unit Tests (`SecureModePluginTest`):**
- Mock environment variables with `MockedStatic<Utils>`
- Create temporary files in `build/mock/` directories
- Test each control independently
- Test valid configurations that should pass
- Test environment variable vs properties file

**Test Pattern:**
```java
@Test(expectedExceptions = IllegalConfigurationException.class)
void testNewControl() throws IOException, ModelException {
    mockSecurityEnv(
        SecureModeUtils.NEW_CONTROL,
        TEST_MODEL_DIR.resolve("file.txt"),
        "content"
    );
}
```

### Common Patterns

**Checking Properties:**
```java
String value = prop.getProperty("option.key");
if (value != null && !isAllowed(value)) {
    throw new IllegalConfigurationException("Message");
}
```

**Checking Environment Variables:**
```java
String value = Utils.getenv("ENV_VAR", Utils.getenv("FALLBACK"));
if (value != null && !isAllowed(value)) {
    throw new IllegalConfigurationException("Message");
}
```

**File Scanning:**
```java
try (Stream<Path> stream = Files.walk(dir)) {
    for (Path p : stream.collect(Collectors.toList())) {
        if (isForbidden(p)) {
            throw new IllegalConfigurationException("Message");
        }
    }
}
```

## Integration Points

### Model Loading Flow

```
ModelServer.registerModel()
    ↓
EventManager.notifyListeners(MODEL_CONFIGURED)
    ↓
SecureModeModelServerListener.onModelConfigured()
    ↓
SecureModeUtils.validateSecurity()
    ↓
[Validation passes] → Model loads
[Validation fails] → IllegalConfigurationException → Model rejected
```

### Adapter Loading Flow

```
Adapter.newInstance() [COMMON PATH for both static and dynamic]
    ↓
Adapter.validateAdapterSecurity()
    ↓
[Validation passes] → Adapter created → register() → modelInfo.registerAdapter()
[Validation fails] → IllegalConfigurationException → Adapter NOT created → NOT in LIST API
```

**Static Adapters (Workflow Config):**
```
AdapterWorkflowFunction.prepare()
    ↓
Adapter.newInstance() → validateAdapterSecurity()
    ↓
[Fails] → Workflow initialization fails → Model doesn't load
```

**Dynamic Adapters (API):**
```
POST /models/{model}/adapters
    ↓
AdapterManagementRequestHandler.handleRegisterAdapter()
    ↓
Adapter.newInstance() → validateAdapterSecurity()
    ↓
[Fails] → HTTP error response → Adapter NOT registered
```

### Engine Filtering

Only Python and MPI engines are validated:
```java
String engine = modelInfo.getEngineName();
if (!"Python".equals(engine) && !"MPI".equals(engine)) {
    logger.info("Skipping security check for engine: {}", engine);
    return;
}
```

### SageMaker Platform

- SageMaker sets environment variables in container
- Plugin automatically activates when `SAGEMAKER_SECURE_MODE=true`
- No user configuration required
- Validation happens before model serves any requests

## Troubleshooting

### Common Issues

**"Security Controls environment variable is not set"**
- `SAGEMAKER_SECURITY_CONTROLS` must be set when secure mode is enabled
- SageMaker platform should set this automatically

**"Property X is prohibited from being set in Secure Mode"**
- Property not in allowlist
- Add to `PROPERTIES_ALLOWLIST` if it should be allowed

**"Custom entrypoint is prohibited in Secure Mode"**
- Entrypoint must start with `djl_python.`
- Check `option.entryPoint`, `DJL_ENTRY_POINT`, `OPTION_ENTRYPOINT`
- Ensure no `model.py` file exists

**"Pickle-based files found in directory"**
- Convert model to Safetensors format
- Remove .bin, .pt, .pth, .ckpt, .pkl files from untrusted directories

### Debugging

Enable debug logging:
```properties
# log4j2.xml
<Logger name="ai.djl.serving.plugins.securemode" level="debug"/>
```

Check environment variables:
```bash
echo $SAGEMAKER_SECURE_MODE
echo $SAGEMAKER_SECURITY_CONTROLS
echo $SAGEMAKER_UNTRUSTED_CHANNELS
```

## Performance Considerations

- File scanning is recursive and may be slow for large directories
- Validation runs once per model at load time (not per request)
- Properties validation is fast (Set lookup)
- JSON parsing only for `tokenizer_config.json` files

## Security Best Practices

1. **Minimize allowlist** - Only add properties that are truly needed
2. **Validate early** - Fail fast before model loads
3. **Clear error messages** - Help users understand what's blocked and why
4. **Test thoroughly** - Each control should have positive and negative tests
5. **Document changes** - Update README when modifying allowlist or controls

## Adapter Security (NEW)

### Implementation

Adapter validation is implemented in `wlm/src/main/java/ai/djl/serving/wlm/Adapter.java`:

```java
public static <I, O> Adapter<I, O> newInstance(...) {
    // Validate adapter security FIRST
    validateAdapterSecurity(src);
    
    // Then create adapter
    return new PyAdapter(...);
}

private static void validateAdapterSecurity(String adapterPath) {
    // Check for model.py, requirements.txt, pickle files
}
```

### What Gets Validated

When adapters are loaded (statically or dynamically):

1. **DISALLOW_CUSTOM_INFERENCE_SCRIPTS** - Blocks `model.py` in adapter directory
2. **DISALLOW_REQUIREMENTS_TXT** - Blocks `requirements.txt` in adapter directory
3. **DISALLOW_PICKLE_FILES** - Blocks `.bin`, `.pt`, `.pth`, `.ckpt`, `.pkl` files (recursively)

### Testing

Comprehensive tests in `SecureModeAdapterTest.java`:
- Adapter with `model.py` → blocked
- Adapter with `requirements.txt` → blocked
- Adapter with pickle files → blocked
- Adapter with `.safetensors` → allowed
- Multiple violations → fails on first
- Secure mode disabled → all allowed

### Key Points

- **Single validation point**: `Adapter.newInstance()` is the common path for all adapter loading
- **Fail fast**: Validation happens before adapter object is created
- **Hard fail**: Exception propagates, adapter never registered
- **Clean state**: Failed adapters don't appear in LIST API

## Related Code

- `wlm/src/main/java/ai/djl/serving/wlm/Adapter.java` - Adapter creation and validation
- `wlm/src/main/java/ai/djl/serving/wlm/ModelInfo.java` - Model and adapter registry
- `serving/src/main/java/ai/djl/serving/http/AdapterManagementRequestHandler.java` - Dynamic adapter API
- `serving/src/main/java/ai/djl/serving/workflow/function/AdapterWorkflowFunction.java` - Static adapter loading
- `serving/src/main/java/ai/djl/serving/wlm/util/EventManager.java` - Event system
- `serving/src/main/java/ai/djl/serving/http/IllegalConfigurationException.java` - Exception type
- `engines/python/setup/djl_python/adapter_manager_mixin.py` - Python adapter management
- `engines/python/` - Python engine that secure mode validates

## References

- Plugin system: `serving/docs/plugin_management.md`
- Model configuration: `serving/docs/configuration.md`
- SageMaker integration: AWS SageMaker documentation
