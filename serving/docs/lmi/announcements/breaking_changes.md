# LMI Breaking Changes

This document details breaking changes for LMI container version releases.

## LMI Container v15 (0.33.0)

### Removal of Sequence Scheduler and LMI-Dist Engines

We have removed the sequence scheduler and lmi dist rolling batch implementations in this release due to most users leveraging vLLM and TensorRT-LLM.

#### Sequence Scheduler Deprecation

If you are using `option.rolling_batch=scheduler`, this deprecation impacts you.
The Sequence Scheduler Rolling Batch enabled continuous batching behavior using the HuggingFace Transformers Library.
This implementation is no longer needed as vLLM supports the same set of models that Sequence Scheduler did with better performance.
To migrate from the sequence scheduler rolling batch, we recommend you move to vLLM by specifying `option.rolling_batch=vllm`.
More details on the vLLM rolling batch implementation can be found in the [vllm user guide](../user_guides/vllm_user_guide.md).

If you still require using the Sequence Scheduler implementation, you can use LMI v14.

#### LMI-Dist Deprecation

If you are using `option.rolling_batch=lmi-dist`, this deprecation impacts you.
The LMI-Dist Rolling Batch has not been updated since LMI v12. 
We are deprecating LMI-Dist in favor of vLLM.
We expect that existing users of lmi-dist can easily transfer to vLLM, as LMI-Dist was largely based on vLLM.

To migrate from lmi-dist to vLLM, you must take the following actions:

* disable MPI mode by setting `engine=Python` and `option.mpi_mode=false`
* swap rolling batch to vllm by setting `option.rolling_batch=vllm`

You can also explore the new async mode support for vLLM [here](../user_guides/vllm_user_guide.md#async-mode-configurations).

We have tested this migration process for models and use-cases in our LMI-Dist Integration tests.
The following use-cases in LMI-Dist will not work with vLLM:

* using Tensor Parallel>1 and Pipeline Parallel>1 in addition to draft model speculative decoding
* multi-node inference

If neither of those use-cases apply to you, we expect a seamless transition to vLLM.