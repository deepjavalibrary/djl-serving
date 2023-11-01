#!/bin/bash
#set -e

verlte() {
    [ "$1" = "$2" ] && return 1 || [ "$2" = "`echo -e "$1\n$2" | sort -V | head -n1`" ]
}

# Follow https://docs.aws.amazon.com/sagemaker/latest/dg/inference-gpu-drivers.html
if [ -f /usr/local/cuda/compat/libcuda.so.1 ]; then
    cat /usr/local/cuda/version.txt
    CUDA_COMPAT_MAX_DRIVER_VERSION=$(readlink /usr/local/cuda/compat/libcuda.so.1 |cut -d'.' -f 3-)
    echo "CUDA compat package requires Nvidia driver ⩽${CUDA_COMPAT_MAX_DRIVER_VERSION}"
    NVIDIA_DRIVER_VERSION=$(sed -n 's/^NVRM.*Kernel Module *\([0-9.]*\).*$/\1/p' /proc/driver/nvidia/version 2>/dev/null || true)
    echo "Current installed Nvidia driver version is ${NVIDIA_DRIVER_VERSION}"
    if [ $(verlte $CUDA_COMPAT_MAX_DRIVER_VERSION $NVIDIA_DRIVER_VERSION) ]; then
        echo "Setup CUDA compatibility libs path to LD_LIBRARY_PATH"
        export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH
        echo $LD_LIBRARY_PATH
    else
        echo "Skip CUDA compat libs setup as newer Nvidia driver is installed"
    fi
else
    echo "Skip CUDA compat libs setup as package not found"
fi

if [[ "$1" = "serve" ]]; then
    shift 1
    code=77
    while [[ code -eq 77 ]]
    do
        /usr/bin/djl-serving "$@"
        code=$?
    done
elif [[ "$1" = "partition" ]] || [[ "$1" = "train" ]]; then
    shift 1
    /usr/bin/python3 /opt/djl/partition/partition.py "$@"
else
    eval "$@"
fi
