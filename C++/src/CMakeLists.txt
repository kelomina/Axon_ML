cmake_minimum_required(VERSION 3.20)
project(kvd LANGUAGES C CXX)

set(KVD_SOURCES
  api.cpp
  config.cpp
  features_byte_sequence.cpp
  features_file_attributes.cpp
  features_pe.cpp
  features_statistics.cpp
  lightgbm_infer.cpp
  malware_scanner.cpp
  family_classifier.cpp
  sha256.cpp
  util_filesystem.cpp
  kvd.def
)

add_library(axon_engine SHARED)
target_sources(axon_engine PRIVATE ${KVD_SOURCES})
target_include_directories(axon_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

add_library(signature_engine SHARED)
target_sources(signature_engine PRIVATE ${KVD_SOURCES})
target_include_directories(signature_engine PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)

find_package(nlohmann_json CONFIG REQUIRED)
find_package(LIEF CONFIG REQUIRED)

find_path(LIGHTGBM_INCLUDE_DIR NAMES LightGBM/c_api.h)
find_library(LIGHTGBM_LIBRARY_RELEASE NAMES lib_lightgbm PATH_SUFFIXES lib)
find_library(LIGHTGBM_LIBRARY_DEBUG NAMES lib_lightgbm PATH_SUFFIXES debug/lib)

add_library(lightgbm::lightgbm UNKNOWN IMPORTED)
set_target_properties(lightgbm::lightgbm PROPERTIES
  IMPORTED_LOCATION_RELEASE "${LIGHTGBM_LIBRARY_RELEASE}"
  IMPORTED_LOCATION_DEBUG "${LIGHTGBM_LIBRARY_DEBUG}"
  INTERFACE_INCLUDE_DIRECTORIES "${LIGHTGBM_INCLUDE_DIR}"
)

target_link_libraries(axon_engine
  PUBLIC
    nlohmann_json::nlohmann_json
    LIEF::LIEF
    lightgbm::lightgbm
    bcrypt
)
target_link_libraries(signature_engine
  PUBLIC
    nlohmann_json::nlohmann_json
    LIEF::LIEF
    lightgbm::lightgbm
    bcrypt
)

target_compile_definitions(axon_engine PRIVATE KVD_BUILD_DLL=1 KVD_BUILD_AXON_ENGINE=1)
target_compile_definitions(signature_engine PRIVATE KVD_BUILD_DLL=1 KVD_BUILD_SIGNATURE_ENGINE=1)

set(AXON_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Axon")
set(SIG_OUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../Signature")

set_target_properties(axon_engine PROPERTIES
  OUTPUT_NAME "axon_engine"
  RUNTIME_OUTPUT_DIRECTORY "${AXON_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${AXON_OUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${AXON_OUT_DIR}"
)
set_target_properties(signature_engine PROPERTIES
  OUTPUT_NAME "signature_engine"
  RUNTIME_OUTPUT_DIRECTORY "${SIG_OUT_DIR}"
  LIBRARY_OUTPUT_DIRECTORY "${SIG_OUT_DIR}"
  ARCHIVE_OUTPUT_DIRECTORY "${SIG_OUT_DIR}"
)

if(WIN32)
  target_link_options(axon_engine PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/kvd.def")
  target_link_options(signature_engine PRIVATE "/DEF:${CMAKE_CURRENT_SOURCE_DIR}/kvd.def")
endif()
