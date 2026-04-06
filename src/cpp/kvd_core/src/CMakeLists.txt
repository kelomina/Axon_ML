cmake_minimum_required(VERSION 3.24)
project(kvd LANGUAGES C CXX)

set(KVD_SOURCES
  api.cpp
  config.cpp
  features_byte_sequence.cpp
  features_file_attributes.cpp
  features_pe.cpp
  features_statistics.cpp
  lightgbm_infer.cpp
  onnx_infer.cpp
  malware_scanner.cpp
  family_classifier.cpp
  sha256.cpp
  util_filesystem.cpp
)

if(WIN32)
  list(APPEND KVD_SOURCES kvd.def)
endif()

add_library(axon_engine SHARED)
target_sources(axon_engine PRIVATE ${KVD_SOURCES})
target_include_directories(axon_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_library(signature_engine SHARED)
target_sources(signature_engine PRIVATE ${KVD_SOURCES})
target_include_directories(signature_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(LIEF CONFIG REQUIRED)

# Replace hardcoded ONNXRuntime path with find_package or fetch_content or user provided path
# Here we use an option with a generic default or expect the user to provide ONNXRUNTIME_ROOT
if(NOT DEFINED ONNXRUNTIME_ROOT)
  if(WIN32)
    set(ONNXRUNTIME_ROOT "C:/onnxruntime" CACHE PATH "Path to ONNX Runtime")
  else()
    set(ONNXRUNTIME_ROOT "/usr/local" CACHE PATH "Path to ONNX Runtime")
  endif()
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")

if(WIN32)
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
  set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/onnxruntime.dll")
else()
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  set(ONNXRUNTIME_DLL "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
endif()

add_library(onnxruntime::onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION_RELEASE "${ONNXRUNTIME_LIBRARY}"
  IMPORTED_IMPLIB_RELEASE "${ONNXRUNTIME_LIBRARY}"
  IMPORTED_LOCATION_DEBUG "${ONNXRUNTIME_LIBRARY}"
  IMPORTED_IMPLIB_DEBUG "${ONNXRUNTIME_LIBRARY}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

if(WIN32)
  add_library(onnxruntime_providers_shared SHARED IMPORTED)
  set_target_properties(onnxruntime_providers_shared PROPERTIES
    IMPORTED_LOCATION_RELEASE "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    IMPORTED_IMPLIB_RELEASE "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib"
    IMPORTED_LOCATION_DEBUG "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.dll"
    IMPORTED_IMPLIB_DEBUG "${ONNXRUNTIME_ROOT}/lib/onnxruntime_providers_shared.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
endif()

if(EXISTS "${ONNXRUNTIME_DLL}")
  file(COPY "${ONNXRUNTIME_DLL}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endif()

find_path(LIGHTGBM_INCLUDE_DIR NAMES LightGBM/c_api.h)
find_library(LIGHTGBM_LIBRARY_RELEASE NAMES lightgbm lib_lightgbm PATH_SUFFIXES lib)
find_library(LIGHTGBM_LIBRARY_DEBUG NAMES lightgbm lib_lightgbm PATH_SUFFIXES debug/lib)

add_library(lightgbm::lightgbm UNKNOWN IMPORTED)
set_target_properties(lightgbm::lightgbm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${LIGHTGBM_LIBRARY_RELEASE}"
  IMPORTED_LOCATION_DEBUG "${LIGHTGBM_LIBRARY_DEBUG}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIGHTGBM_INCLUDE_DIR}"
)

set(KVD_LINK_LIBS
  nlohmann_json::nlohmann_json
  LIEF::LIEF
  lightgbm::lightgbm
  onnxruntime::onnxruntime
)
if(WIN32)
  list(APPEND KVD_LINK_LIBS onnxruntime_providers_shared bcrypt)
endif()

target_link_libraries(axon_engine PUBLIC ${KVD_LINK_LIBS})
target_link_libraries(signature_engine PUBLIC ${KVD_LINK_LIBS})

target_compile_definitions(axon_engine PRIVATE KVD_BUILD_DLL=1 KVD_BUILD_AXON_ENGINE=1)
target_compile_definitions(signature_engine PRIVATE KVD_BUILD_DLL=1 KVD_BUILD_SIGNATURE_ENGINE=1)

set_target_properties(axon_engine PROPERTIES OUTPUT_NAME "axon_engine")
set_target_properties(signature_engine PROPERTIES OUTPUT_NAME "signature_engine")

if(WIN32)
  target_link_options(axon_engine PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/kvd.def")
  target_link_options(signature_engine PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/kvd.def")
endif()
