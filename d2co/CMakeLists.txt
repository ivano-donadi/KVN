project( d2co )

cmake_minimum_required(VERSION 3.9)

message("** Project ${PROJECT_NAME} **")

set(VERSION_MAJOR   0   CACHE STRING "Project major version number.")
set(VERSION_MINOR   1   CACHE STRING "Project minor version number.")
set(VERSION_PATCH   0   CACHE STRING "Project patch version number.")
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)

# Setup the default build environment
SET(CMAKE_AR "gcc-ar")
SET(CMAKE_NM "gcc-nm")
SET(CMAKE_RANLIB "gcc-ranlib")
SET(C "gcc")
SET(CXX "g++")

aux_source_directory( src D2CO_SRC )
add_library(d2co STATIC ${D2CO_SRC})

add_dependencies( d2co cv_ext)


list( APPEND CXX_CUSTOM_FLAGS -std=c++11 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -ffast-math -fPIC )
list( APPEND CUDA_CUSTOM_FLAGS -std=c++11 --use_fast_math --relocatable-device-code=true )

if(NOT CMAKE_BUILD_TYPE)
    message("Build type automatically set to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" )
else(NOT CMAKE_BUILD_TYPE)
    message("Build type: '${CMAKE_BUILD_TYPE}'")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "WARNING: you will get poor performance!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

option(USE_SSE "Build with SSE4+ support" ON)
option(USE_AVX "Build with AVX/AVX2 support" OFF)
option(USE_NEON "Build with NEON support" OFF)
option(USE_CUDA "Build with CUDA support" OFF)
#option(USE_CAFFE_TRIPLET "Build with Caffe Triplet CNN support" OFF)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("Compiling for x86 / x86_64 architectures")

    list(APPEND CXX_CUSTOM_FLAGS -march=nocona)

    if(USE_AVX)
        message("AVX/AVX2 support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC D2CO_USE_AVX)
        list(APPEND CXX_CUSTOM_FLAGS -mavx2 -mfma -march=core-avx2)
    elseif(USE_SSE)
        message("SSE4+ support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC D2CO_USE_SSE)
        list(APPEND CXX_CUSTOM_FLAGS -msse4.1)
    endif()

    if(USE_CUDA)
        enable_language(CUDA)
        target_compile_definitions(${PROJECT_NAME} PUBLIC DUMMY_PROJECT_USE_CUDA)
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)")
    message("Compiling for ARM v7 architectures")

    list(APPEND CXX_CUSTOM_FLAGS -mcpu=cortex-a53 -mfloat-abi=hard)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC D2CO_USE_NEON)
        list(APPEND CXX_CUSTOM_FLAGS -mfpu=neon-fp-armv8 -mneon-for-64bits)
    endif(USE_NEON)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
    message("Compiling for aarch64 architectures")

    list(APPEND CXX_CUSTOM_FLAGS -march=armv8-a+crypto -mcpu=cortex-a57+crypto)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC D2CO_USE_NEON)
    endif(USE_NEON)

else()
    message(WARNING "Unsupported architecture: using default compiler flags")
endif ()

find_package(OpenMP REQUIRED)
list(APPEND CXX_CUSTOM_FLAGS ${OpenMP_CXX_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# The previous line should be replaced in the future with:
# target_link_options(${PROJECT_NAME} PUBLIC  ${OpenMP_EXE_LINKER_FLAGS})

# Finally, set the compilers optpions
target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)
if(USE_CUDA)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_CUSTOM_FLAGS}>)
endif()


find_package( OpenCV REQUIRED )
# This is to be complinat with OpenCV >= 3.3.1
target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )

find_package(Eigen3 REQUIRED)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

find_package(Ceres  REQUIRED)

find_library( DIME_LIBS dime )
find_library( GLEW_LIBS GLEW )
find_library( OPENMESHCORE_LIBS OpenMeshCore PATHS "/usr/local/lib/OpenMesh" )
find_package( Boost COMPONENTS program_options filesystem system REQUIRED )

if(USE_CAFFE_TRIPLET)
  find_library(Caffe_LIBS NAMES caffe HINTS /opt/caffe_triplet/lib)
  include_directories(/opt/caffe_triplet/include)
endif(USE_CAFFE_TRIPLET)                      

target_include_directories( ${PROJECT_NAME} PUBLIC
                            $<INSTALL_INTERFACE:include>
                            $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                            ${OpenCV_INCLUDE_DIRS}
                            ${Boost_INCLUDE_DIRS}
                            ${EIGEN3_INCLUDE_DIR}
                            ${CERES_INCLUDE_DIRS}
                            ${OPENGL_INCLUDE_DIRS}
                            ${GLFW_INCLUDE_DIRS})

target_link_libraries( d2co PUBLIC
                       cv_ext
                       ${OpenMP_CXX_LIBRARIES}
                       ${Boost_LIBRARIES}
                       ${DIME_LIBS}
                       ${OpenCV_LIBS}
                       ${CERES_LIBRARIES}
                       ${OPENGL_LIBRARY}
                       ${GLEW_LIBS}
                       ${GLFW_STATIC_LIBRARIES}
                       ${OPENMESHCORE_LIBS})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


set_target_properties(d2co PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set_target_properties(d2co PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})


 add_executable(test_localization apps/test_localization.cpp apps/apps_utils.cpp )
 target_link_libraries( test_localization
                        d2co)
 set_target_properties( test_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(test_model apps/test_model.cpp apps/apps_utils.cpp )
target_link_libraries( test_model d2co)
set_target_properties( test_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(eval_linemod_dataset apps/eval_linemod_dataset.cpp apps/apps_utils.cpp )
target_link_libraries( eval_linemod_dataset d2co )
set_target_properties( eval_linemod_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# add_executable(test_distance_maps apps/test_distance_maps.cpp apps/apps_utils.cpp )
# target_link_libraries( test_distance_maps
#                        d2co)
# set_target_properties( test_distance_maps PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)


 add_executable(generate_templates apps/generate_templates.cpp apps/apps_utils.cpp )
 target_link_libraries( generate_templates
                        d2co)
 set_target_properties( generate_templates PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

 add_executable(obj_localization apps/obj_localization.cpp apps/apps_utils.cpp )
 target_link_libraries( obj_localization
                        d2co)
 set_target_properties( obj_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(stereo_obj_localization apps/stereo_obj_localization.cpp apps/apps_utils.cpp )
target_link_libraries( stereo_obj_localization d2co)
set_target_properties( stereo_obj_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# add_executable(test_automatica apps/test_automatica.cpp apps/apps_utils.cpp apps/automatica_localization.cpp )
# target_link_libraries( test_automatica
#                        d2co)
# set_target_properties( test_automatica PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if(USE_CAFFE_TRIPLET)

  add_executable(sphereview_data apps/sphereview_data.cpp )
  target_link_libraries( sphereview_data
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( sphereview_data PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

  add_executable(cnn_3dobj_classify apps/cnn_3dobj_classify.cpp )
  target_link_libraries( cnn_3dobj_classify
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( cnn_3dobj_classify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

  add_executable(test_cnn_triplet apps/test_cnn_triplet.cpp )
  target_link_libraries( test_cnn_triplet
                         d2co
                         ${Caffe_LIBS})
  set_target_properties( test_cnn_triplet PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
  
endif(USE_CAFFE_TRIPLET)   


add_executable(test_object_templates_generator apps/test_object_templates_generator.cpp apps/apps_utils.cpp )
target_link_libraries( test_object_templates_generator d2co)
set_target_properties( test_object_templates_generator PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

if ( NOT TARGET cv_ext )
    add_subdirectory(cv_ext )
endif( NOT TARGET cv_ext )