# cmake/add_executorch.cmake
#
# Locates and integrates Meta's ExecuTorch C++ runtime and hardware delegates
# (CoreML, MLX/MPS, XNNPACK, Optimized CPU Kernels) for NetBender (.pte models).
#
# Automatically discovers ExecuTorch from:
#   1. The project's Python virtual environment (.venv / $VIRTUAL_ENV / CONDA_ENV_PATH)
#      installed via `pip install -r requirements.txt` (`executorch>=1.0.0`)
#   2. Or a custom CMake prefix via `-DEXECUTORCH_ROOT=/path/to/executorch`

option(ENABLE_EXECUTORCH_BACKEND "Enable ExecuTorch (.pte) inference backend" ON)

if (NOT ENABLE_EXECUTORCH_BACKEND)
  set(EXECUTORCH_FOUND FALSE)
  return()
endif()

if (NOT DEFINED EXECUTORCH_ROOT)
  if (DEFINED ENV{EXECUTORCH_ROOT})
    set(EXECUTORCH_ROOT "$ENV{EXECUTORCH_ROOT}")
  else()
    # Prefer the repository's .venv or active VIRTUAL_ENV / CONDA_ENV_PATH Python interpreter
    set(_et_python_candidates "")
    if (EXISTS "${CMAKE_SOURCE_DIR}/../.venv/bin/python")
      list(APPEND _et_python_candidates "${CMAKE_SOURCE_DIR}/../.venv/bin/python")
    endif()
    if (DEFINED ENV{VIRTUAL_ENV} AND EXISTS "$ENV{VIRTUAL_ENV}/bin/python")
      list(APPEND _et_python_candidates "$ENV{VIRTUAL_ENV}/bin/python")
    endif()
    if (DEFINED CONDA_ENV_PATH AND EXISTS "${CONDA_ENV_PATH}/bin/python")
      list(APPEND _et_python_candidates "${CONDA_ENV_PATH}/bin/python")
    endif()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if (Python3_Interpreter_FOUND)
      list(APPEND _et_python_candidates "${Python3_EXECUTABLE}")
    endif()

    foreach(_py_exec IN LISTS _et_python_candidates)
      execute_process(
        COMMAND "${_py_exec}" -c "import executorch, os; print(os.path.dirname(executorch.__file__))"
        OUTPUT_VARIABLE _py_et_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
      )
      if (_py_et_dir AND EXISTS "${_py_et_dir}/include")
        set(EXECUTORCH_ROOT "${_py_et_dir}")
        break()
      elseif (_py_et_dir AND EXISTS "${_py_et_dir}/data/include")
        set(EXECUTORCH_ROOT "${_py_et_dir}/data")
        break()
      endif()
    endforeach()

    # Fallback: glob .venv/lib/python*/site-packages/executorch directly
    if (NOT EXECUTORCH_ROOT)
      file(GLOB _venv_et_dirs "${CMAKE_SOURCE_DIR}/../.venv/lib/python*/site-packages/executorch")
      foreach(_cand IN LISTS _venv_et_dirs)
        if (EXISTS "${_cand}/include")
          set(EXECUTORCH_ROOT "${_cand}")
          break()
        endif()
      endforeach()
    endif()
  endif()
endif()

find_path(EXECUTORCH_INCLUDE_DIR
  NAMES executorch/extension/module/module.h
  PATHS
    "${EXECUTORCH_ROOT}/include"
    "${CMAKE_SOURCE_DIR}/../executorch/include"
  NO_DEFAULT_PATH
)

find_library(EXECUTORCH_CORE_LIB
  NAMES executorch executorch_core
  PATHS
    "${EXECUTORCH_ROOT}/lib"
    "${CMAKE_SOURCE_DIR}/../executorch/lib"
  NO_DEFAULT_PATH
)

if (EXECUTORCH_INCLUDE_DIR AND EXECUTORCH_CORE_LIB)
  set(EXECUTORCH_FOUND TRUE)
  get_filename_component(EXECUTORCH_LIBRARY_DIR "${EXECUTORCH_CORE_LIB}" DIRECTORY)
  message(STATUS "ExecuTorch C++ Runtime found: ${EXECUTORCH_CORE_LIB}")

  set(EXECUTORCH_INCLUDE_DIRS "${EXECUTORCH_INCLUDE_DIR}")
  if (EXISTS "${EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10")
    list(APPEND EXECUTORCH_INCLUDE_DIRS "${EXECUTORCH_INCLUDE_DIR}/executorch/runtime/core/portable_type/c10")
  endif()

  set(EXECUTORCH_COMPILE_DEFINITIONS
    "NETBENDER_HAS_EXECUTORCH=1"
    "C10_USING_CUSTOM_GENERATED_MACROS"
    "ET_EVENT_TRACER_ENABLED"
  )

  set(EXECUTORCH_LIBRARIES "${EXECUTORCH_CORE_LIB}")

  # Discover prebuilt shared components (from executorch>=1.0 wheel) or static libraries
  foreach(et_lib_name
      executorch_kernels_optimized
      executorch_kernels_torchao
      executorch_backend_coreml
      executorch_backend_xnnpack
      executorch_backend_mlx
      executorch_threadpool
      extension_module
      extension_module_static
      extension_tensor
      extension_data_loader
      extension_flat_tensor
      portable_ops_lib
      portable_kernels
      optimized_native_cpu_ops_lib
      xnnpack_backend
      XNNPACK
      pthreadpool
      cpuinfo
      coremldelegate
      mpsdelegate)
    find_library(ET_OPT_LIB_${et_lib_name}
      NAMES ${et_lib_name}
      PATHS
        "${EXECUTORCH_ROOT}/lib"
        "${CMAKE_SOURCE_DIR}/../executorch/lib"
      NO_DEFAULT_PATH
    )
    if (ET_OPT_LIB_${et_lib_name})
      list(APPEND EXECUTORCH_LIBRARIES "${ET_OPT_LIB_${et_lib_name}}")
      if (et_lib_name STREQUAL "executorch_threadpool")
        list(APPEND EXECUTORCH_COMPILE_DEFINITIONS "ET_USE_THREADPOOL")
      endif()
    endif()
  endforeach()

  if (APPLE)
    list(APPEND EXECUTORCH_LIBRARIES
      "-Wl,-rpath,${EXECUTORCH_LIBRARY_DIR}"
      "-framework Foundation"
      "-framework CoreML"
      "-framework Accelerate"
      "-framework Metal"
      "-framework MetalPerformanceShaders"
      "-framework MetalPerformanceShadersGraph"
    )
  elseif (UNIX)
    list(APPEND EXECUTORCH_LIBRARIES "-Wl,-rpath,${EXECUTORCH_LIBRARY_DIR}")
  endif()
else()
  set(EXECUTORCH_FOUND FALSE)
  message(STATUS "ExecuTorch C++ libraries not found. Install executorch in .venv (`pip install -r requirements.txt`) or pass -DEXECUTORCH_ROOT=/path/to/executorch.")
endif()

