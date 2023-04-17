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


int main(int argc, char** argv)
{
  std::string model_repository_path;
  int verbose_level = 0;

  // Parse commandline...
  int opt;
  while ((opt = getopt(argc, argv, "vm:r:")) != -1) {
    switch (opt) {
      case 'm': {
        enforce_memory_type = true;
        if (!strcmp(optarg, "system")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU;
        } else if (!strcmp(optarg, "pinned")) {
          requested_memory_type = TRITONSERVER_MEMORY_CPU_PINNED;
        } else if (!strcmp(optarg, "gpu")) {
          requested_memory_type = TRITONSERVER_MEMORY_GPU;
        } else {
        }
        break;
      }
      case 'r':
        model_repository_path = optarg;
        break;
      case 'v':
        verbose_level = 1;
        break;
      case '?':
        break;
    }
  }

  if (model_repository_path.empty()) {
    Usage(argv, "-r must be used to specify model repository path");
  }

  // Check API version. This compares the API version of the
  // triton-server library linked into this application against the
  // API version of the header file used when compiling this
  // application. The API version of the shared library must be >= the
  // API version used when compiling this application.
  uint32_t api_version_major, api_version_minor;
  FAIL_IF_ERR(
      TRITONSERVER_ApiVersion(&api_version_major, &api_version_minor),
      "getting Triton API version");
  if ((TRITONSERVER_API_VERSION_MAJOR != api_version_major) ||
      (TRITONSERVER_API_VERSION_MINOR > api_version_minor)) {
    FAIL("triton server API version mismatch");
  }

  // Create the option setting to use when creating the inference
  // server object.
  TRITONSERVER_ServerOptions* server_options = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsNew(&server_options),
      "creating server options");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetModelRepositoryPath(
          server_options, model_repository_path.c_str()),
      "setting model repository path");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetLogVerbose(server_options, verbose_level),
      "setting verbose logging level");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetBackendDirectory(
          server_options, "/opt/tritonserver/backends"),
      "setting backend directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetRepoAgentDirectory(
          server_options, "/opt/tritonserver/repoagents"),
      "setting repository agent directory");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetStrictModelConfig(server_options, true),
      "setting strict model configuration");
  double min_compute_capability = 0;
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsSetMinSupportedComputeCapability(
          server_options, min_compute_capability),
      "setting minimum supported CUDA compute capability");

  // Create the server object using the option settings. The server
  // object encapsulates all the functionality of the Triton server
  // and allows access to the Triton server API. Typically only a
  // single server object is needed by an application, but it is
  // allowed to create multiple server objects within a single
  // application. After the server object is created the server
  // options can be deleted.
  TRITONSERVER_Server* server_ptr = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ServerNew(&server_ptr, server_options),
      "creating server object");
  FAIL_IF_ERR(
      TRITONSERVER_ServerOptionsDelete(server_options),
      "deleting server options");

  // Use a shared_ptr to manage the lifetime of the server object.
  std::shared_ptr<TRITONSERVER_Server> server(
      server_ptr, TRITONSERVER_ServerDelete);

  // Wait until the server is both live and ready. The server will not
  // appear "ready" until all models are loaded and ready to receive
  // inference requests.
  size_t health_iters = 0;
  while (true) {
    bool live, ready;
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsLive(server.get(), &live),
        "unable to get server liveness");
    FAIL_IF_ERR(
        TRITONSERVER_ServerIsReady(server.get(), &ready),
        "unable to get server readiness");
    std::cout << "Server Health: live " << live << ", ready " << ready
              << std::endl;
    if (live && ready) {
      break;
    }

    if (++health_iters >= 10) {
      FAIL("failed to find healthy inference server");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(500));
  }

  // Server metadata can be accessed using the server object. The
  // metadata is returned as an abstract TRITONSERVER_Message that can
  // be converted to JSON for further processing.
  {
    TRITONSERVER_Message* server_metadata_message;
    FAIL_IF_ERR(
        TRITONSERVER_ServerMetadata(server.get(), &server_metadata_message),
        "unable to get server metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(
            server_metadata_message, &buffer, &byte_size),
        "unable to serialize server metadata message");

    std::cout << "Server Metadata:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;

    FAIL_IF_ERR(
        TRITONSERVER_MessageDelete(server_metadata_message),
        "deleting server metadata message");
  }

  const std::string model_name("fastertransformer");

  // We already waited for the server to be ready, above, so we know
  // that all models are also ready. But as an example we also wait
  // for a specific model to become available.
  bool is_ready = false;
  health_iters = 0;
  rapidjson::Document model_metadata;
  while (!is_ready) {
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelIsReady(
            server.get(), model_name.c_str(), 1 /* model_version */, &is_ready),
        "unable to get model readiness");
    if (!is_ready) {
      if (++health_iters >= 10) {
        FAIL("model failed to be ready in 10 iterations");
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
      continue;
    }

    TRITONSERVER_Message* model_metadata_message;
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(
            server.get(), model_name.c_str(), 1, &model_metadata_message),
        "unable to get model metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(
            model_metadata_message, &buffer, &byte_size),
        "unable to serialize model metadata");

    // Parse the JSON string that represents the model metadata into a
    // JSON document. We use rapidjson for this parsing but any JSON
    // parser can be used.
    model_metadata.Parse(buffer, byte_size);
    if (model_metadata.HasParseError()) {
      FAIL(
          "error: failed to parse model metadata from JSON: " +
          std::string(GetParseError_En(model_metadata.GetParseError())) +
          " at " + std::to_string(model_metadata.GetErrorOffset()));
    }

    FAIL_IF_ERR(
        TRITONSERVER_MessageDelete(model_metadata_message),
        "deleting model metadata message");

    // Now that we have a document representation of the model
    // metadata, we can query it to extract some information about the
    // model.
    if (strcmp(model_metadata["name"].GetString(), model_name.c_str())) {
      FAIL("unable to find metadata for model");
    }

    bool found_version = false;
    if (model_metadata.HasMember("versions")) {
      for (const auto& version : model_metadata["versions"].GetArray()) {
        if (strcmp(version.GetString(), "1") == 0) {
          found_version = true;
          break;
        }
      }
    }
    if (!found_version) {
      FAIL("unable to find version 1 status for model");
    }
  }

  // When triton needs a buffer to hold an output tensor, it will ask
  // us to provide the buffer. In this way we can have any buffer
  // management and sharing strategy that we want. To communicate to
  // triton the functions that we want it to call to perform the
  // allocations, we create a "response allocator" object. We pass
  // this response allocate object to triton when requesting
  // inference. We can reuse this response allocate object for any
  // number of inference requests.
  TRITONSERVER_ResponseAllocator* allocator = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorNew(
          &allocator, ResponseAlloc, ResponseRelease, nullptr /* start_fn */),
      "creating response allocator");

  // Create an inference request object. The inference request object
  // is where we set the name of the model we want to use for
  // inference and the input tensors.
  TRITONSERVER_InferenceRequest* irequest = nullptr;
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestNew(
          &irequest, server.get(), model_name.c_str(), -1 /* model_version */),
      "creating inference request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetId(irequest, "my_request_id"),
      "setting ID for the request");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestSetReleaseCallback(
          irequest, InferRequestComplete, nullptr /* request_release_userp */),
      "setting request release callback");


  uint32_t inputids[] = {13959, 1566, 12, 2968, 10, 2059, 18, 476, 2741, 18, 667, 40, 26, 8528, 18961, 9651, 44, 1051, 5901, 13552, 1};

  auto input_ids = "input_ids";
  auto sequence_length = "sequence_length";
  auto max_output_len = "max_output_len";
  std::vector<int64_t> input_ids_shape({1, sizeof(inputids)/sizeof(inputids[0])});
  std::vector<int64_t> sequence_length_shape({1, 1});
  std::vector<int64_t> max_output_len_shape({1, 1});

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, input_ids, TRITONSERVER_TYPE_UINT32,
          &input_ids_shape[0], input_ids_shape.size()),
      "setting input_idsmeta-data for the request");
  std::vector<char> input_ids_data;
  input_ids_data.resize(sizeof(inputids)/sizeof(inputids[0]) * sizeof(uint32_t));
  for (size_t i = 0; i < sizeof(inputids)/sizeof(inputids[0]); i++) {
    ((uint32_t*)input_ids_data.data())[i] = inputids[i];
  }
  FAIL_IF_ERR(
    TRITONSERVER_InferenceRequestAppendInputData(
        irequest, input_ids, &input_ids_data[0], input_ids_data.size(), requested_memory_type,
        0 /* memory_type_id */),
  "assigning input_ids data");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, sequence_length, TRITONSERVER_TYPE_UINT32,
          &sequence_length_shape[0], sequence_length_shape.size()),
      "setting sequence_length meta-data for the request");
  std::vector<char> sequence_length_data;
  sequence_length_data.resize(sizeof(uint32_t));
  ((uint32_t*)sequence_length_data.data())[0] = 21;
  FAIL_IF_ERR(
    TRITONSERVER_InferenceRequestAppendInputData(
        irequest, sequence_length, &sequence_length_data[0], sequence_length_data.size(), requested_memory_type,
        0 /* memory_type_id */),
  "assigning sequence_length data");

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddInput(
          irequest, max_output_len, TRITONSERVER_TYPE_UINT32,
          &max_output_len_shape[0], max_output_len_shape.size()),
      "setting max_output_len meta-data for the request");
  std::vector<char> max_output_len_data;
  max_output_len_data.resize(sizeof(uint32_t));
  ((uint32_t*)max_output_len_data.data())[0] = 128;
  FAIL_IF_ERR(
    TRITONSERVER_InferenceRequestAppendInputData(
        irequest, max_output_len, &max_output_len_data[0], max_output_len_data.size(), requested_memory_type,
        0 /* memory_type_id */),
  "assigning max_output_len data");


  auto output_ids = "output_ids";

  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, output_ids),
      "requesting output_ids for the request");
  auto out_sequence_length = "sequence_length";
  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestAddRequestedOutput(irequest, out_sequence_length),
      "requesting sequence_length for the request");


  // Perform inference by calling TRITONSERVER_ServerInferAsync. This
  // call is asychronous and therefore returns immediately. The
  // completion of the inference and delivery of the response is done
  // by triton by calling the "response complete" callback functions
  // (InferResponseComplete in this case).
  {
    auto p = new std::promise<TRITONSERVER_InferenceResponse*>();
    std::future<TRITONSERVER_InferenceResponse*> completed = p->get_future();

    FAIL_IF_ERR(
        TRITONSERVER_InferenceRequestSetResponseCallback(
            irequest, allocator, nullptr /* response_allocator_userp */,
            InferResponseComplete, reinterpret_cast<void*>(p)),
        "setting response callback");

    FAIL_IF_ERR(
        TRITONSERVER_ServerInferAsync(
            server.get(), irequest, nullptr /* trace */),
        "running inference");

    // The InferResponseComplete function sets the std::promise so
    // that this thread will block until the response is returned.
    TRITONSERVER_InferenceResponse* completed_response = completed.get();


    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseError(completed_response),
        "response status");

    Check(
        completed_response, output_ids, out_sequence_length, TRITONSERVER_TYPE_INT32);

    FAIL_IF_ERR(
        TRITONSERVER_InferenceResponseDelete(completed_response),
        "deleting inference response");
  }


  FAIL_IF_ERR(
      TRITONSERVER_InferenceRequestDelete(irequest),
      "deleting inference request");

  FAIL_IF_ERR(
      TRITONSERVER_ResponseAllocatorDelete(allocator),
      "deleting response allocator");

  return 0;
}
