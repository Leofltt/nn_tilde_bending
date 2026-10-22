# cmake/add_onnxruntime.cmake
#
# Downloads and integrates the official Microsoft ONNX Runtime C++ release
# for macOS (Apple Silicon / Intel) and Windows.

set(ort_dir ${CMAKE_CURRENT_BINARY_DIR}/../onnxruntime)
set(ort_version "1.18.0")

if (APPLE)
  set(ort_archive_name "onnxruntime-osx-universal2-${ort_version}.tgz")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-osx-universal2-${ort_version}")
elseif (WIN32)
  set(ort_archive_name "onnxruntime-win-x64-${ort_version}.zip")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-win-x64-${ort_version}")
else() # Linux
  set(ort_archive_name "onnxruntime-linux-x64-${ort_version}.tgz")
  set(ort_url "https://github.com/microsoft/onnxruntime/releases/download/v${ort_version}/${ort_archive_name}")
  set(ort_extracted_dir "${ort_dir}/onnxruntime-linux-x64-${ort_version}")
endif()

# Check if ONNX Runtime is already downloaded
if (NOT EXISTS "${ort_extracted_dir}/include/onnxruntime_cxx_api.h")
  message(STATUS "Fetching pre-built ONNX Runtime v${ort_version}...")
  file(MAKE_DIRECTORY "${ort_dir}")
  
  set(ort_archive_file "${ort_dir}/${ort_archive_name}")
  if (NOT EXISTS "${ort_archive_file}")
    message(STATUS "Downloading ${ort_url} to ${ort_archive_file}...")
    file(DOWNLOAD "${ort_url}" "${ort_archive_file}" SHOW_PROGRESS)
  endif()

  message(STATUS "Extracting ${ort_archive_file}...")
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar -xf "${ort_archive_file}"
    WORKING_DIRECTORY "${ort_dir}"
  )
endif()

if (EXISTS "${ort_extracted_dir}/include/onnxruntime_cxx_api.h")
  set(ONNXRUNTIME_FOUND TRUE)
  set(ONNXRUNTIME_INCLUDE_DIR "${ort_extracted_dir}/include")
  
  if (APPLE)
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/libonnxruntime.dylib")
  elseif (WIN32)
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/onnxruntime.lib")
    set(ONNXRUNTIME_DLL "${ort_extracted_dir}/lib/onnxruntime.dll")
  else()
    set(ONNXRUNTIME_LIBRARY "${ort_extracted_dir}/lib/libonnxruntime.so")
  endif()
  
  message(STATUS "ONNX Runtime located at: ${ort_extracted_dir}")
  message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
  message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

  # Create imported target for clean linkage
  add_library(onnxruntime SHARED IMPORTED GLOBAL)
  set_target_properties(onnxruntime PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
  )
  if (WIN32)
    set_target_properties(onnxruntime PROPERTIES
      IMPORTED_IMPLIB "${ONNXRUNTIME_LIBRARY}"
    )
  endif()
else()
  message(WARNING "Failed to locate ONNX Runtime C++ headers in ${ort_extracted_dir}")
endif()
