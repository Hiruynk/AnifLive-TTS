cmake_minimum_required(VERSION 3.26)

project(aniflive_tts_mamba2_update_plugin LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(TENSORRT_INCLUDE_DIR "" CACHE PATH "TensorRT public include directory")
set(TENSORRT_LIBRARY "" CACHE FILEPATH "TensorRT nvinfer shared library")

if(NOT EXISTS "${TENSORRT_INCLUDE_DIR}/NvInfer.h")
  message(FATAL_ERROR "TENSORRT_INCLUDE_DIR must contain NvInfer.h")
endif()

if(NOT EXISTS "${TENSORRT_LIBRARY}")
  message(FATAL_ERROR "TENSORRT_LIBRARY must point to libnvinfer")
endif()

find_package(CUDAToolkit 12.8 REQUIRED)

add_library(
  aniflive_mamba2_update_plugin
  SHARED
  src/mamba2_update_kernel.cu
  src/mamba2_update_plugin.cpp
)

target_include_directories(
  aniflive_mamba2_update_plugin
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

target_include_directories(
  aniflive_mamba2_update_plugin
  SYSTEM PRIVATE
  "${TENSORRT_INCLUDE_DIR}"
)

target_link_libraries(
  aniflive_mamba2_update_plugin
  PRIVATE
  "${TENSORRT_LIBRARY}"
  CUDA::cudart
)

target_compile_options(
  aniflive_mamba2_update_plugin
  PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
)

set_target_properties(
  aniflive_mamba2_update_plugin
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CUDA_SEPARABLE_COMPILATION ON
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  OUTPUT_NAME "aniflive_mamba2_update_plugin"
)
