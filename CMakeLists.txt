cmake_minimum_required(VERSION 3.18...3.31)
project(spheropack LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(SPHEROPACK_BUILD_TESTS "Build the C++ unit tests (doctest)" OFF)

# Header-only core.
add_library(spheropack_core INTERFACE)
target_include_directories(spheropack_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(spheropack_core INTERFACE cxx_std_20)
# No fused multiply-add contraction: keeps event prediction bit-identical across
# compilers for a given platform.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spheropack_core INTERFACE -ffp-contract=off)
elseif(MSVC)
  target_compile_options(spheropack_core INTERFACE /fp:precise)
endif()

if(SKBUILD)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule)
  find_package(nanobind CONFIG REQUIRED)
  nanobind_add_module(_core STABLE_ABI NB_STATIC src/bindings.cpp)
  target_link_libraries(_core PRIVATE spheropack_core)
  install(TARGETS _core LIBRARY DESTINATION spheropack)
endif()

if(SPHEROPACK_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests/cpp)
endif()
