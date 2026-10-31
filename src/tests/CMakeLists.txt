cmake_minimum_required(VERSION 3.15)
project(nn_bending_tests)

add_executable(test_backends
    test_backends.cpp
)

target_include_directories(test_backends PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_SOURCE_DIR}/backend"
    "${CMAKE_SOURCE_DIR}/shared"
    "${ONNXRUNTIME_INCLUDE_DIR}"
)

target_compile_features(test_backends PRIVATE cxx_std_20)

target_link_libraries(test_backends PRIVATE
    backend
    "${TORCH_LIBRARIES}"
    onnxruntime
)

if (APPLE)
    set_target_properties(test_backends PROPERTIES
        BUILD_WITH_INSTALL_RPATH TRUE
        INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib;${torch_dir}/libtorch/lib;${ort_extracted_dir}/lib;${CONDA_ENV_PATH}/lib;/Library/Application Support/NetBender/lib;@loader_path/;@loader_path/../lib"
    )
endif()
