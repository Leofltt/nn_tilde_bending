cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(nn_tilde)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

configure_file(
  "${CMAKE_SOURCE_DIR}/../install/max-linker-flags.txt" "${CMAKE_SOURCE_DIR}/frontend/maxmsp/min-api/max-sdk-base/script/max-linker-flags.txt"
  COPYONLY
)

if (NOT DEFINED SIGN_ID)
  if (DEFINED $ENV{SIGN_ID})
    set(SIGN_ID $ENV{SIGN_ID})
  else()
    set(SIGN_ID "-")
  endif()
endif()

message("Copying ${CMAKE_SOURCE_DIR}/../install/MaxAPI.lib" "${CMAKE_SOURCE_DIR}/frontend/maxmsp/min-api/max-sdk-base/c74support/max-includes/x64" )
configure_file(
    "${CMAKE_SOURCE_DIR}/../install/MaxAPI.lib" "${CMAKE_SOURCE_DIR}/frontend/maxmsp/min-api/max-sdk-base/c74support/max-includes/x64/MaxAPI.lib" 
    COPYONLY
)

configure_file(
  "${CMAKE_SOURCE_DIR}/../install/patch_with_vst.sh" "${CMAKE_SOURCE_DIR}/extras"
  COPYONLY
)


include(${CMAKE_SOURCE_DIR}/cmake/add_torch.cmake)
list(PREPEND CMAKE_PREFIX_PATH "${torch_dir}/libtorch")
find_package(Torch REQUIRED PATHS ${torch_dir}/libtorch/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(CONDA_ENV_PATH "${CMAKE_SOURCE_DIR}/../env")
if (MSVC)
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}:${CONDA_ENV_PATH}")
else()
  set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};${CONDA_ENV_PATH}")
endif()

find_package(Python3 COMPONENTS Interpreter QUIET)
if (Python3_Interpreter_FOUND)
  set(PYTHON_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python interpreter")
else()
  set(PYTHON_EXECUTABLE "${CONDA_ENV_PATH}/bin/python" CACHE FILEPATH "Python interpreter")
endif()

find_package(CURL QUIET)
if (CURL_FOUND)
  set(CURL_INCLUDE_DIR "${CURL_INCLUDE_DIRS}" CACHE PATH "CURL include directory")
  set(CURL_LIBRARY "${CURL_LIBRARIES}" CACHE FILEPATH "CURL library")
else()
  if (UNIX)
    set(CURL_INCLUDE_DIR "${CONDA_ENV_PATH}/include" CACHE PATH "CURL include directory")
    if (APPLE)
      set(CURL_LIBRARY "${CONDA_ENV_PATH}/lib/libcurl.dylib" CACHE FILEPATH "CURL library")
    else()
      set(CURL_LIBRARY "${CONDA_ENV_PATH}/lib/libcurl.so" CACHE FILEPATH "CURL library")
    endif()
  elseif(MSVC)
    set(VCPKG_PATH "${CMAKE_SOURCE_DIR}/../vcpkg")
    set(CURL_INCLUDE_DIR "${VCPKG_PATH}/packages/curl_x64-windows/include" CACHE PATH "CURL include directory")
    set(CURL_LIBRARY "${VCPKG_PATH}/packages/curl_x64-windows/lib/libcurl.lib" CACHE FILEPATH "CURL library")
  endif()
endif()


if (APPLE)
  set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(SET CMP00076)


# import json
add_subdirectory(json)

if (UNIX)
  if (APPLE)
    add_compile_options(-std=c++20)
    set(CMAKE_CXX_FLAGS "-faligned-allocation")
    if (CMAKE_OSX_ARCHITECTURES STREQUAL "")
        set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR})
    endif()
    message("Building for architecture : ${CMAKE_OSX_ARCHITECTURES} ")
  endif()
endif()



add_subdirectory(backend) # DEEP LEARNING BACKEND

execute_process(
		COMMAND git describe --tags
		WORKING_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}
		OUTPUT_VARIABLE VERSION
		OUTPUT_STRIP_TRAILING_WHITESPACE
)

option(BUILD_MAXMSP "Build MaxMSP targets" ON)
option(BUILD_PUREDATA "Build PureData targets" ON)

if (NOT DEFINED NO_PUREDATA)
  set(NO_PUREDATA 0)
endif()

if (BUILD_PUREDATA AND NO_PUREDATA EQUAL 0)
  if ("${PUREDATA_INCLUDE_DIR}" STREQUAL "")
    set(PUREDATA_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pd_include")
    execute_process(
      COMMAND cmake -E make_directory "${PUREDATA_INCLUDE_DIR}"
    )
    file(DOWNLOAD "https://raw.githubusercontent.com/pure-data/pure-data/master/src/m_pd.h" "${PUREDATA_INCLUDE_DIR}/m_pd.h")
  endif()
  add_subdirectory(frontend/puredata/nn_tilde) # PURE DATA EXTERNAL
else()
  if (UNIX AND NOT APPLE) 
    message(FATAL_ERROR "NO_PUREDATA needs to be off on Linux, otherwise no task are available")
  endif()
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../package-info.json.in "${CMAKE_CURRENT_SOURCE_DIR}/package-info.json")

if(BUILD_MAXMSP AND (APPLE OR MSVC))
    # MAX MSP EXTERNAL
    add_subdirectory(frontend/maxmsp/nn.info) 
    add_subdirectory(frontend/maxmsp/nn_tilde)
    add_subdirectory(frontend/maxmsp/mc.nn_tilde)
    add_subdirectory(frontend/maxmsp/mcs.nn_tilde)
endif()

if (BUILD_MAXMSP AND (APPLE OR MSVC))
    add_custom_target(maxmsp)
    add_dependencies(maxmsp nn_tilde mc.nn_tilde mcs.nn_tilde nn.info)
endif()

if (BUILD_PUREDATA AND NO_PUREDATA EQUAL 0)
    add_custom_target(puredata)
    add_dependencies(puredata nn)
endif()
