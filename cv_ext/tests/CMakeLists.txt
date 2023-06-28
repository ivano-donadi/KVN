file(GLOB_RECURSE TESTS "*.cpp")

enable_testing()
find_package(GTest REQUIRED)
include_directories(. ${GTEST_INCLUDE_DIRS})

aux_source_directory( . CV_EXT_TESTS_SRC )

add_executable( cv_ext_tests ${CV_EXT_TESTS_SRC} )
target_link_libraries( cv_ext_tests cv_ext ${GTEST_BOTH_LIBRARIES})
set_target_properties( cv_ext_tests PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
