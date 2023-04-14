/*
 * Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

#ifndef DJL_TRITON_AI_DJL_TRITON_HELPER_H
#define DJL_TRITON_AI_DJL_TRITON_HELPER_H
#include <jni.h>

extern jclass ENGINE_EXCEPTION_CLASS;
static constexpr const jint RELEASE_MODE = JNI_ABORT;

#define DJL_CHECK_WITH_MSG(cond, error_msg)           \
  if (!cond) {                                        \
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, error_msg); \
  }

/*
 * Macros to guard beginning and end section of all functions
 * every function starts with API_BEGIN()
 * and finishes with API_END()
 */
#define API_BEGIN() try {
#define API_END()                                                      \
  }                                                                    \
  catch (const std::exception& e_) {                                   \
    env->ThrowNew(ENGINE_EXCEPTION_CLASS, e_.what());                  \
  }

// TODO refactor all jni functions to c style function which mean
//  return value should be unified to function execution status code
#define API_END_RETURN() \
  API_END()              \
  return 0;

inline TRITONSERVER_Error* ResponseAlloc(
        TRITONSERVER_ResponseAllocator* allocator, const char* tensor_name,
        size_t byte_size, TRITONSERVER_MemoryType preferred_memory_type,
        int64_t preferred_memory_type_id, void* userp, void** buffer,
        void** buffer_userp, TRITONSERVER_MemoryType* actual_memory_type,
        int64_t* actual_memory_type_id) {
    // Initially attempt to make the actual memory type and id that we
    // allocate be the same as preferred memory type
    *actual_memory_type = preferred_memory_type;
    *actual_memory_type_id = preferred_memory_type_id;

    // If 'byte_size' is zero just return 'buffer' == nullptr, we don't
    // need to do any other book-keeping.
    if (byte_size == 0) {
        *buffer = nullptr;
        *buffer_userp = nullptr;
        std::cout << "allocated " << byte_size << " bytes for result tensor "
                  << tensor_name << std::endl;
    } else {
        void *allocated_ptr = nullptr;

        switch (*actual_memory_type) {
            // Use CPU memory if the requested memory type is unknown
            // (default case).
            case TRITONSERVER_MEMORY_CPU:
            default: {
                *actual_memory_type = TRITONSERVER_MEMORY_CPU;
                allocated_ptr = malloc(byte_size);
                break;
            }
        }

        // Pass the tensor name with buffer_userp so we can show it when
        // releasing the buffer.
        if (allocated_ptr != nullptr) {
            *buffer = allocated_ptr;
            *buffer_userp = new std::string(tensor_name);
            std::cout << "allocated " << byte_size << " bytes in "
                      << TRITONSERVER_MemoryTypeString(*actual_memory_type)
                      << " for result tensor " << tensor_name << std::endl;
        }
    }
    return nullptr; // Success
}

inline TRITONSERVER_Error* ResponseRelease(
        TRITONSERVER_ResponseAllocator* allocator, void* buffer, void* buffer_userp,
        size_t byte_size, TRITONSERVER_MemoryType memory_type,
        int64_t memory_type_id)
{
    std::string* name = nullptr;
    if (buffer_userp != nullptr) {
        name = reinterpret_cast<std::string*>(buffer_userp);
    } else {
        name = new std::string("<unknown>");
    }

    std::cout << "Releasing buffer " << buffer << " of size " << byte_size
              << " in " << TRITONSERVER_MemoryTypeString(memory_type)
              << " for result '" << *name << "'" << std::endl;
    switch (memory_type) {
        case TRITONSERVER_MEMORY_CPU:
            free(buffer);
            break;
        default:
            std::cerr << "error: unexpected buffer allocated in CUDA managed memory"
                      << std::endl;
            break;
    }

    delete name;

    return nullptr;  // Success
}

inline void InferRequestComplete(
        TRITONSERVER_InferenceRequest* request, const uint32_t flags, void* userp)
{
    // We reuse the request so we don't delete it here.
}

inline void InferResponseComplete(
        TRITONSERVER_InferenceResponse* response, const uint32_t flags, void* userp)
{
    if (response != nullptr) {
        // Send 'response' to the future.
        std::promise<TRITONSERVER_InferenceResponse*>* p =
                reinterpret_cast<std::promise<TRITONSERVER_InferenceResponse*>*>(userp);
        p->set_value(response);
        delete p;
    }
}


#endif //DJL_TRITON_AI_DJL_TRITON_HELPER_H
