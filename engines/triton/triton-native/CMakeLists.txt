cmake_minimum_required (VERSION 3.18)
project(djl_triton LANGUAGES C CXX)

set(JAVA_AWT_LIBRARY NotNeeded)
set(JAVA_AWT_INCLUDE_PATH NotNeeded)
find_package(JNI REQUIRED)

set(SOURCE_FILES
         src/main/native/ai_djl_triton_jni_TritonLibrary.cc
        src/main/native/ai_djl_triton_helper.h)

add_library(djl_triton SHARED ${SOURCE_FILES})

target_compile_features(djl_triton PRIVATE cxx_std_11)

target_compile_options(
    djl_triton
    PRIVATE
    -Wall -Wextra -Wno-type-limits -Wno-unused-parameter -Wno-deprecated-declarations -Werror
)

set(CMAKE_SKIP_RPATH TRUE)
set_target_properties(djl_triton PROPERTIES LINK_FLAGS "-Wl,-rpath,$ORIGIN")

target_include_directories(
djl_triton
PRIVATE
    ${JNI_INCLUDE_DIRS}
    tritonserver/include
    build/include
)

target_link_libraries(
djl_triton
PRIVATE
    /opt/tritonserver/lib/libtritonserver.so
)

