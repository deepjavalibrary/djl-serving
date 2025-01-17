# **Troubleshooting Guide**

This guide provides steps and information to troubleshoot issues related to the model server and debugging tools. It is a work in progress and will eventually be moved to `serving/docs` upon finalization.

---

## **Profiling**

> Note that profiling is still being worked upon and the interfaces are bound to change until finalized. In it's current state this is only recommended for personal debugging.

The container can be started in **DEBUG mode** by setting the environment variable `DEBUG_MODE=1`. When enabled, this mode facilitates advanced profiling and debugging capabilities with the following effects:

### **1. Installation of Debugging Tools**

In DEBUG mode, the following tool will be installed automatically:

- **[NVIDIA Nsight Systems](https://docs.nvidia.com/nsight-systems/)**
  - Nsight Systems enables system-wide performance analysis.
  - The version of Nsight can be controlled using the environment variable:
    - `NSIGHT_VERSION`: Specifies the version of Nsight Systems to install (e.g., `2024.6.1`).

### **2. Profiling with Nsight Systems**

The model server will automatically start under the `nsys` profiler when `DEBUG_MODE` is enabled. The following environment variables can be configured to customize the profiling behavior:

- **`NSYS_PROFILE_DELAY`**:  
  - Specifies the delay in seconds before profiling begins.  
  - Use this to exclude startup activities and capture only relevant information.  
  - **Default**: `30` seconds.

- **`NSYS_PROFILE_DURATION`**:  
  - Specifies the duration in seconds for profiling.  
  - Avoid setting this to values larger than 600 seconds (10 minutes) to prevent generating large and unwieldy reports.  
  - **Default**: `600` seconds.

- **`NSYS_PROFILE_TRACE`**:  
  - Allows customization of the APIs and operations to trace.  
  - Examples include `cuda`, `nvtx`, `osrt`, `cudnn`, `cublas`, `mpi`, and `python-gil`.  
  - Refer to the [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) for more details.

### **3. Report Generation and Upload**

- After profiling is complete, the generated `.nsys-rep` report will be automatically uploaded to the specified S3 bucket if the `S3_DEBUG_PATH` environment variable is provided.  
- **`S3_DEBUG_PATH`**:  
  - Specifies the S3 bucket and path for storing the profiling report.  
  - **Example**: `s3://my-bucket/profiles/`.

---

### **Example Usage**

To enable profiling and customize its behavior:

```bash
DEBUG_MODE=1 \
NSIGHT_VERSION=2024.6.1 \
NSYS_PROFILE_DELAY=20 \
NSYS_PROFILE_DURATION=300 \
NSYS_PROFILE_TRACE="cuda,nvtx,osrt" \
S3_DEBUG_PATH="s3://my-bucket/debug-reports/" \
docker run my-container
