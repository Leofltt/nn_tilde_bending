cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(backend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(backend STATIC parsing_utils.cpp backend.cpp OnnxBackend.cpp)
target_include_directories(backend PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(backend "${TORCH_LIBRARIES}" onnxruntime)
set_property(TARGET backend PROPERTY CXX_STANDARD 20)




