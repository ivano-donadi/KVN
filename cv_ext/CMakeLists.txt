project( cv_ext )

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

aux_source_directory( src CV_EXT_SRC )
add_library(${PROJECT_NAME} STATIC ${CV_EXT_SRC})
add_library(iterative_pnp_stereo MODULE src/iterative_pnp_stereo.cpp src/pinhole_camera_model.cpp src/serialization.cpp)
add_library(iterative_pnp_stereo_variance MODULE src/iterative_pnp_stereo_variance.cpp src/pinhole_camera_model.cpp src/serialization.cpp)
add_library(iterative_pnp_stereo_prob MODULE src/iterative_pnp_stereo_prob.cpp src/pinhole_camera_model.cpp src/serialization.cpp)



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

option(BUILD_TESTS "Build tests" ON)
option(BUILD_EXAMPLES "Build examples" OFF)
option(GENERATE_DOCS "Generate documentation" OFF)

option(USE_PCL "Build with PCL (Point Cloud Library) support" OFF)

option(USE_SSE "Build with SSE4+ support" ON)
option(USE_AVX "Build with AVX/AVX2 support" OFF)
option(USE_NEON "Build with NEON support" OFF)
option(USE_CUDA "Build with CUDA support" OFF)

if(USE_PCL)
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_PCL)
endif()


if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("Compiling for x86 / x86_64 architectures")

    list(APPEND CXX_CUSTOM_FLAGS -march=nocona)

    if(USE_AVX)
        message("AVX/AVX2 support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_AVX)
        list(APPEND CXX_CUSTOM_FLAGS -mavx2 -mfma -march=core-avx2)
    elseif(USE_SSE)
        message("SSE4+ support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_SSE)
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
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_NEON)
        list(APPEND CXX_CUSTOM_FLAGS -mfpu=neon-fp-armv8 -mneon-for-64bits)
    endif(USE_NEON)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
    message("Compiling for aarch64 architectures")

    list(APPEND CXX_CUSTOM_FLAGS -march=armv8-a+crypto -mcpu=cortex-a57+crypto)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USE_NEON)
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
target_compile_options(iterative_pnp_stereo PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)
target_compile_options(iterative_pnp_stereo_variance PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)
if(USE_CUDA)
    target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_CUSTOM_FLAGS}>)
endif()

# If you want to force a specific version of openCV, say OpenCV-4.5.1, compiled from source
# and installed on a custom folder with the CMAKE_INSTALL_PREFIX cmake flag, say
# /usr/local/OpenCV-4.5.1 , you should set the OpenCV_DIR variable before find_package(), e.g.:
# SET(OpenCV_DIR /usr/local/OpenCV-4.5.1/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )
# This is to be complinat with OpenCV >= 3.3.1
target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )
target_compile_definitions(iterative_pnp_stereo PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )
target_compile_definitions(iterative_pnp_stereo_variance PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )

if( "${OpenCV_VERSION_MAJOR}" STREQUAL "3" )
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USING_OPENCV3)
    target_compile_definitions(iterative_pnp_stereo PUBLIC CV_EXT_USING_OPENCV3)
    target_compile_definitions(iterative_pnp_stereo_variance PUBLIC CV_EXT_USING_OPENCV3)
elseif( "${OpenCV_VERSION_MAJOR}" STREQUAL "4" )
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USING_OPENCV4)
    target_compile_definitions(iterative_pnp_stereo PUBLIC CV_EXT_USING_OPENCV4)
    target_compile_definitions(iterative_pnp_stereo_variance PUBLIC CV_EXT_USING_OPENCV4)
endif()

find_package( Eigen3 REQUIRED )

if(USE_PCL)
    find_package( PCL REQUIRED )
    add_definitions(${PCL_DEFINITIONS})

    # Workaround to avoit a QT5 bug: https://gitlab.kitware.com/cmake/cmake/issues/16915
    if(NOT "${PCL_LIBRARIES}" STREQUAL "")
        list(REMOVE_ITEM PCL_LIBRARIES
                "vtkGUISupportQt"
                "vtkGUISupportQtOpenGL"
                "vtkGUISupportQtSQL"
                "vtkGUISupportQtWebkit"
                "vtkViewsQt"
                "vtkRenderingQt")
    endif()
endif()

find_package( Boost COMPONENTS program_options filesystem system python numpy REQUIRED )
find_package(PythonLibs 3.6 REQUIRED)
find_package( Ceres REQUIRED)
find_library( LIBUUID_LIBRARIES uuid)

# Lib yaml-cpp
find_package(yaml-cpp REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

target_link_libraries( ${PROJECT_NAME} PUBLIC
        ${OpenMP_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${LIBUUID_LIBRARIES}
        ${YAML_CPP_LIBRARIES})

target_link_libraries( iterative_pnp_stereo PUBLIC
        ${OpenMP_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${LIBUUID_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${PYTHON_LIBRARIES})

target_link_libraries( iterative_pnp_stereo_variance PUBLIC
        ${OpenMP_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${LIBUUID_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${PYTHON_LIBRARIES})

target_link_libraries( iterative_pnp_stereo_prob PUBLIC
        ${OpenMP_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${LIBUUID_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        ${PYTHON_LIBRARIES})



target_include_directories( ${PROJECT_NAME} PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR})

target_include_directories( iterative_pnp_stereo PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS})

target_include_directories( iterative_pnp_stereo_prob PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS})

target_include_directories( iterative_pnp_stereo_variance PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        ${CERES_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${YAML_CPP_INCLUDE_DIR}
        ${PYTHON_INCLUDE_DIRS})

if(USE_PCL)
    target_link_libraries( ${PROJECT_NAME} PUBLIC ${PCL_LIBRARIES})
    target_include_directories( ${PROJECT_NAME} PUBLIC ${PCL_INCLUDE_DIRS})
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set_target_properties(iterative_pnp_stereo PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set_target_properties(iterative_pnp_stereo_prob PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set_target_properties(iterative_pnp_stereo_variance PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

if( BUILD_TESTS )
    add_subdirectory(tests)
endif( BUILD_TESTS )

if( BUILD_EXAMPLES )
    add_subdirectory(examples)
endif( BUILD_EXAMPLES )

if( GENERATE_DOCS )
    add_subdirectory(doc)
endif( GENERATE_DOCS )
