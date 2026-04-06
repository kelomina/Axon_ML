cmake_minimum_required(VERSION 3.24)

project(KoloVirusDetectorNative VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KVD_BUILD_TESTS "Build tests" ON)
option(KVD_BUILD_EXAMPLES "Build examples" ON)
option(KVD_USE_ASAN "Enable AddressSanitizer" OFF)

if(KVD_USE_ASAN)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang|GNU")
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
  elseif(MSVC)
    add_compile_options(/fsanitize=address)
  endif()
endif()

add_subdirectory(src)

if(KVD_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(KVD_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
