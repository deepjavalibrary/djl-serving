# OutofMemory handling in djl-serving

This document explains properties that can be configured in djl-serving to better handle OutOfMemory exceptions.

The following properties can be configured in `serving.properties` file per each model

* `required_memory_mb`: Required memory for CPU and GPU in MB to load the model. GPU required memory can be overridden by setting `gpu.required_memory_mb`.
* `gpu.required_memory_mb`: Required GPU memory in MB to load the model. This allows user to set a different value for GPU required memory from CPU required memory. If this is not specified, `required_memory_mb` will be used for GPU as well if specified.
* `reserved_memory_mb`: Memory to reserve in MB in addition to required memory to account for inference memory costs for CPU and GPU.
* `gpu.reserved_memory_mb`: GPU memory to reserve in MB in addition to required memory to account for inference memory costs. This allows user to set a different value for GPU reserved memory from CPU reserved memory. If this is not specified, `reserved_memory_mb` will be used for GPU as well if specified.


djl-serving will use `required_memory_mb`  and `reserved_memory_mb` to decide whether a model can be loaded and successful inference request can run. djl-serving will fetch free memory available on CPU and GPU and check whether free memory is greater than `required_memory_mb` plus `reserved_memory_mb` . If djl-serving cannot load the model due to inadequate free memory, it throws HTTP `507` error facilitating clients to handle the error for e.g by unloading few models and re-trying.

This approach helps us with:

* Failing fast without needing to download the model from an external repository
* Prevents the need to create backend process and eventually leading to killed process


In addition to user configurable properties, djl-serving’s python engine handles exceptions of types `OutOfMemoryError` (e.g `torch.cuda.OutOfMemoryError`), `MemoryError`  during both load and inference time and returns HTTP `507` error. Out of memory exception handling is best effort from djl-serving and there’s risk of python process getting killed already or the memory cannot freed correctly.
