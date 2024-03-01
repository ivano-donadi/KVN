project( iterative_pnp_stereo )

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


add_library(iterative_pnp_stereo MODULE src/iterative_pnp_stereo.cpp src/pinhole_camera_model.cpp src/serialization.cpp)
add_library(iterative_pnp_stereo_variance MODULE src/iterative_pnp_stereo_variance.cpp src/pinhole_camera_model.cpp src/serialization.cpp)
add_library(iterative_pnp_stereo_prob MODULE src/iterative_pnp_stereo_prob.cpp src/pinhole_camera_model.cpp src/serialization.cpp)


list( APPEND CXX_CUSTOM_FLAGS -std=c++11 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated -O3 -ffast-math -fPIC )

if(NOT CMAKE_BUILD_TYPE)
    message("Build type automatically set to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" )
else(NOT CMAKE_BUILD_TYPE)
    message("Build type: '${CMAKE_BUILD_TYPE}'")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "WARNING: you will get poor performance!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)



# Finally, set the compilers optpions
target_compile_options(iterative_pnp_stereo PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)
target_compile_options(iterative_pnp_stereo_variance PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)
target_compile_options(iterative_pnp_stereo_prob PUBLIC $<$<COMPILE_LANGUAGE:CXX>:${CXX_CUSTOM_FLAGS}>)


# If you want to force a specific version of openCV, say OpenCV-4.5.1, compiled from source
# and installed on a custom folder with the CMAKE_INSTALL_PREFIX cmake flag, say
# /usr/local/OpenCV-4.5.1 , you should set the OpenCV_DIR variable before find_package(), e.g.:
# SET(OpenCV_DIR /usr/local/OpenCV-4.5.1/lib/cmake/opencv4)

find_package( OpenCV REQUIRED )
# This is to be complinat with OpenCV >= 3.3.1
target_compile_definitions(iterative_pnp_stereo PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )
target_compile_definitions(iterative_pnp_stereo_variance PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )
target_compile_definitions(iterative_pnp_stereo_prob PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )

if( "${OpenCV_VERSION_MAJOR}" STREQUAL "3" )
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USING_OPENCV3)
    target_compile_definitions(iterative_pnp_stereo PUBLIC CV_EXT_USING_OPENCV3)
    target_compile_definitions(iterative_pnp_stereo_variance PUBLIC CV_EXT_USING_OPENCV3)
    target_compile_definitions(iterative_pnp_stereo_prob PUBLIC CV_EXT_USING_OPENCV3)
elseif( "${OpenCV_VERSION_MAJOR}" STREQUAL "4" )
    target_compile_definitions(${PROJECT_NAME} PUBLIC CV_EXT_USING_OPENCV4)
    target_compile_definitions(iterative_pnp_stereo PUBLIC CV_EXT_USING_OPENCV4)
    target_compile_definitions(iterative_pnp_stereo_variance PUBLIC CV_EXT_USING_OPENCV4)
    target_compile_definitions(iterative_pnp_stereo_prob PUBLIC CV_EXT_USING_OPENCV4)
endif()

find_package( Eigen3 REQUIRED )

find_package( Boost COMPONENTS program_options filesystem system python numpy REQUIRED )
find_package(PythonLibs 3.6 REQUIRED)
find_package( Ceres REQUIRED)
find_library( LIBUUID_LIBRARIES uuid)

# Lib yaml-cpp
find_package(yaml-cpp REQUIRED)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)



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


set_target_properties(iterative_pnp_stereo PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(iterative_pnp_stereo_prob PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set_target_properties(iterative_pnp_stereo_variance PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
