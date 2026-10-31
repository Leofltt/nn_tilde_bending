cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(backend)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Compile ExecuTorchBackend.cpp in an isolated OBJECT library so ExecuTorch's
# vendored c10 headers and macros never collide with LibTorch's c10 headers.
add_library(executorch_backend_obj OBJECT ExecuTorchBackend.cpp)
set_property(TARGET executorch_backend_obj PROPERTY CXX_STANDARD 20)
set_property(TARGET executorch_backend_obj PROPERTY POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(executorch_backend_obj PRIVATE NETBENDER_NO_TORCH_HEADERS=1)
target_link_libraries(executorch_backend_obj PRIVATE nlohmann_json::nlohmann_json)

if (EXECUTORCH_FOUND)
  target_compile_definitions(executorch_backend_obj PRIVATE ${EXECUTORCH_COMPILE_DEFINITIONS})
  target_include_directories(executorch_backend_obj PRIVATE ${EXECUTORCH_INCLUDE_DIRS})
else()
  target_compile_definitions(executorch_backend_obj PRIVATE NETBENDER_HAS_EXECUTORCH=0)
endif()

add_library(backend STATIC
  parsing_utils.cpp
  backend.cpp
  OnnxBackend.cpp
  $<TARGET_OBJECTS:executorch_backend_obj>
)
target_include_directories(backend PRIVATE "${ONNXRUNTIME_INCLUDE_DIR}")
target_link_libraries(backend "${TORCH_LIBRARIES}" onnxruntime nlohmann_json::nlohmann_json)

if (EXECUTORCH_FOUND)
  target_compile_definitions(backend PUBLIC NETBENDER_HAS_EXECUTORCH=1)
  target_link_libraries(backend ${EXECUTORCH_LIBRARIES})
else()
  target_compile_definitions(backend PUBLIC NETBENDER_HAS_EXECUTORCH=0)
endif()

set_property(TARGET backend PROPERTY CXX_STANDARD 20)




