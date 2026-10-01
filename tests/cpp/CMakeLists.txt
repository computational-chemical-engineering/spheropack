include(FetchContent)
FetchContent_Declare(doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.4.11
  GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(doctest)

add_executable(spheropack_tests test_main.cpp test_rng.cpp test_predict.cpp)
target_link_libraries(spheropack_tests PRIVATE spheropack_core doctest::doctest)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(spheropack_tests PRIVATE -Wall -Wextra -Wpedantic)
endif()
add_test(NAME spheropack_tests COMMAND spheropack_tests)
