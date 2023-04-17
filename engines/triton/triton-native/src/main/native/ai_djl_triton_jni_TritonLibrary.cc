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

#include <rapidjson/document.h>
#include <rapidjson/error/en.h>
#include <unistd.h>
#include <chrono>
#include <cstring>
#include <future>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <jni.h>
#include "tritonserver.h"
#include "ai_djl_triton_jni_TritonLibrary.h"
#include "ai_djl_triton_helper.h"


JNIEXPORT jlong JNICALL Java_ai_djl_triton_jni_TritonLibrary_createServerOption
        (JNIEnv *env, jobject jthis, jstring jrepositoryPath, jstring jbackendsPath,
         jstring jrepoAgentPath, jint verbose) {
    API_BEGIN()
    TRITONSERVER_ServerOptions* server_options_ptr = nullptr;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsNew(&server_options_ptr),
            "creating server options");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetModelRepositoryPath(
                    server_options_ptr, env->GetStringUTFChars(jrepositoryPath,JNI_FALSE)),
            "setting model repository path");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetLogVerbose(server_options_ptr, verbose),
            "setting verbose logging level");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetBackendDirectory(
                    server_options_ptr, env->GetStringUTFChars(jbackendsPath,JNI_FALSE)),
            "setting backend directory");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
                    server_options_ptr, env->GetStringUTFChars(jrepoAgentPath,JNI_FALSE)),
            "setting repository agent directory");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options_ptr, true),
            "setting strict model configuration");
    // Let DJL handle which models to load
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetModelControlMode(
                    server_options_ptr, TRITONSERVER_MODEL_CONTROL_EXPLICIT),
            "setting model control");
    double min_compute_capability = 0;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
                    server_options_ptr, min_compute_capability),
            "setting minimum supported CUDA compute capability");
    return reinterpret_cast<uintptr_t>(server_options_ptr);
    API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_deleteServerOption
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    API_BEGIN()
    auto* server_options_ptr = reinterpret_cast<TRITONSERVER_ServerOptions*>(jhandle);
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsDelete(server_options_ptr), "deleting server options");
    API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_triton_jni_TritonLibrary_createModelServer
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    API_BEGIN()
    auto* server_options_ptr = reinterpret_cast<TRITONSERVER_ServerOptions*>(jhandle);
    TRITONSERVER_Server* server_ptr = nullptr;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerNew(&server_ptr, server_options_ptr),
            "creating server object");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerOptionsDelete(server_options_ptr),
            "deleting server options");
    std::shared_ptr<TRITONSERVER_Server> server(
            server_ptr, TRITONSERVER_ServerDelete);
    return reinterpret_cast<uintptr_t>(server.get());
    API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_deleteModelServer
        (JNIEnv *env, jobject jthis, jlong jhandle) {
    API_BEGIN()
    auto* server_ptr = reinterpret_cast<TRITONSERVER_Server*>(jhandle);
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerDelete(server_ptr), "deleting server object");
    API_END()
}

JNIEXPORT jlong JNICALL Java_ai_djl_triton_jni_TritonLibrary_buildInferenceRequest
        (JNIEnv *env, jobject jthis, jlong jhandle, jstring modelName, jstring id) {
    API_BEGIN()
    auto* server = reinterpret_cast<TRITONSERVER_Server*>(jhandle);
    TRITONSERVER_InferenceRequest* irequest = nullptr;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestNew(&irequest, server,
                     env->GetStringUTFChars(modelName, JNI_FALSE),
                     -1 /* model_version */), "creating inference request");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestSetId(irequest,
                     env->GetStringUTFChars(id, JNI_FALSE)), "setting ID for the request");
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestSetReleaseCallback(irequest, InferRequestComplete,
                     nullptr), "setting request release callback");
    return reinterpret_cast<uintptr_t>(irequest);
    API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_deleteInferenceRequest
        (JNIEnv *env , jobject jthis, jlong irHandle) {
    API_BEGIN()
    auto* irequest = reinterpret_cast<TRITONSERVER_InferenceRequest*>(irHandle);
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestDelete(irequest), "deleting inference request");
    API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_addInput
        (JNIEnv *env, jobject jthis, jlong jir_handle, jstring jname, jobject jbuffer, jlong jbuffer_size, jlongArray jshape, jint jdtype) {
    API_BEGIN()
    auto* irequest = reinterpret_cast<TRITONSERVER_InferenceRequest*>(jir_handle);
    jsize shape_length = env->GetArrayLength(jshape);
    // TODO: potential memory leak
    const auto* jptrs = reinterpret_cast<int64_t*>(env->GetLongArrayElements(jshape, JNI_FALSE));
    const int dtype = jdtype;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestAddInput(irequest, env->GetStringUTFChars(jname, JNI_FALSE), TRITONSERVER_datatype_enum(dtype),
                    jptrs, shape_length), "setting input for the request")
    auto* buffer_address =  env->GetDirectBufferAddress(jbuffer);
    TRITONSERVER_MemoryType requested_memory_type;
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestAppendInputData(
                    irequest, env->GetStringUTFChars(jname, JNI_FALSE), buffer_address, jbuffer_size, requested_memory_type,
                    0 /* memory_type_id */), "assigning input_ids data")
    API_END()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_addOutput
        (JNIEnv *env, jobject jthis, jlong jir_handle, jstring joutput_name) {
    API_BEGIN()
    auto* irequest = reinterpret_cast<TRITONSERVER_InferenceRequest*>(jir_handle);
    DJL_CHECK_WITH_MSG(TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, env->GetStringUTFChars(joutput_name, JNI_FALSE)),
            "requesting output_ids for the request");
    API_END()
}

JNIEXPORT jlongArray JNICALL Java_ai_djl_triton_jni_TritonLibrary_performInference
        (JNIEnv *env, jobject jthis, jlong jserver_handle, jlong jir_handle) {
    API_BEGIN()
    auto* irequest = reinterpret_cast<TRITONSERVER_InferenceRequest*>(jir_handle);
    auto* server_ptr = reinterpret_cast<TRITONSERVER_Server*>(jserver_handle);
    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

    TRITONSERVER_ResponseAllocator* allocator = nullptr;
        DJL_CHECK_WITH_MSG(
                TRITONSERVER_ResponseAllocatorNew(
                        &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
                "creating response allocator");

        DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceRequestSetResponseCallback(
                    irequest, allocator, nullptr /* response_allocator_userp */,
                    InferResponseComplete, reinterpret_cast<void*>(p)),
            "setting response callback");

        DJL_CHECK_WITH_MSG(
            TRITONSERVER_ServerInferAsync(
                    server_ptr, irequest, nullptr /* trace */),
            "running inference");

    // The InferResponseComplete function sets the std::promise so
    // that this thread will block until the response is returned.
    TRITONSERVER_InferenceResponse* completed_response = completed.get();


    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceResponseError(completed_response), "response status")
    jlongArray jarray = env->NewLongArray(2);
    std::vector<jlong> jptrs;
    jptrs.reserve(2);
    jptrs[0] = reinterpret_cast<uintptr_t>(allocator);
    jptrs[1] = reinterpret_cast<uintptr_t>(completed_response);
    env->SetLongArrayRegion(jarray, 0, 2, jptrs.data());
    return jarray;
    API_END_RETURN()
}

JNIEXPORT jobjectArray JNICALL Java_ai_djl_triton_jni_TritonLibrary_fetchResult
        (JNIEnv *env, jobject jthis, jlong jhandler, jobjectArray j2d_shapes, jintArray jdtypes) {
    API_BEGIN()

    auto* response = reinterpret_cast<TRITONSERVER_InferenceResponse*>(jhandler);
    uint32_t output_count;
    DJL_CHECK_WITH_MSG(TRITONSERVER_InferenceResponseOutputCount(response, &output_count),
            "getting number of response outputs")
    uint32_t expected_output_length = env->GetArrayLength(jdtypes);
    if (output_count != expected_output_length) {
        env->ThrowNew(ENGINE_EXCEPTION_CLASS, "Expected output and actual mismatch!");
    }

    jobjectArray result = env->NewObjectArray(output_count, env->FindClass("java/nio/ByteBuffer"), nullptr);

    for (uint32_t idx = 0; idx < output_count; ++idx) {
        const char* cname;
        TRITONSERVER_DataType datatype;
        const int64_t* shape;
        uint64_t dim_count;
        const void* base;
        size_t byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        void* userp;

        DJL_CHECK_WITH_MSG(TRITONSERVER_InferenceResponseOutput(
                        response, idx, &cname, &datatype, &shape, &dim_count, &base,
                        &byte_size, &memory_type, &memory_type_id, &userp),
                "getting output info");

        jlongArray jlong_array = env->NewLongArray(dim_count);
        env->SetLongArrayRegion(jlong_array, 0, dim_count, reinterpret_cast<const jlong*>(shape));
        env->SetObjectArrayElement(j2d_shapes, idx, jlong_array);
        int dtype_intval = datatype;
        env->SetIntArrayRegion(jdtypes, idx, 1, &dtype_intval);

        env->SetObjectArrayElement(result, idx, env->NewDirectByteBuffer(const_cast<void *>(base), byte_size));
    }
    return result;
    API_END_RETURN()
}

JNIEXPORT void JNICALL Java_ai_djl_triton_jni_TritonLibrary_deleteResponse
        (JNIEnv *env, jobject jthis, jlongArray jhandlers) {
    API_BEGIN()
    jlong* jarr = env->GetLongArrayElements(jhandlers, JNI_FALSE);
    auto* allocator = reinterpret_cast<TRITONSERVER_ResponseAllocator*>(jarr[0]);
    auto* completed_response = reinterpret_cast<TRITONSERVER_InferenceResponse*>(jarr[1]);
    DJL_CHECK_WITH_MSG(
            TRITONSERVER_InferenceResponseDelete(completed_response),
            "deleting inference response")
    DJL_CHECK_WITH_MSG(
                TRITONSERVER_ResponseAllocatorDelete(allocator),
                "deleting response allocator")
    env->ReleaseLongArrayElements(jhandlers, jarr, RELEASE_MODE);
    API_END()
}
