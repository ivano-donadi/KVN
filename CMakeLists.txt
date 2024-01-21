# Setup the default build environment
SET(CMAKE_AR "gcc-ar")
SET(CMAKE_NM "gcc-nm")
SET(CMAKE_RANLIB "gcc-ranlib")
SET(C "gcc")
SET(CXX "g++")

aux_source_directory( src OBJ_LOC_SRC )
add_library(${PROJECT_NAME} STATIC ${OBJ_LOC_SRC})

add_dependencies( ${PROJECT_NAME} d2co)
add_dependencies( ${PROJECT_NAME} cv_ext)

target_compile_options(${PROJECT_NAME} PUBLIC -std=c++11 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated 
                                              -pthread -O3 -ffast-math -fPIC)

if(NOT CMAKE_BUILD_TYPE)
    message("Build type automatically set to 'Release'")
    set(CMAKE_BUILD_TYPE "Release" )
else(NOT CMAKE_BUILD_TYPE)
    message("Build type: '${CMAKE_BUILD_TYPE}'")
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        message(WARNING "WARNING: you will get poor performance!")
    endif(CMAKE_BUILD_TYPE STREQUAL "Debug")
endif(NOT CMAKE_BUILD_TYPE)

option(USE_SSE "Build with SSE3+ support" ON)
option(USE_AVX "Build with AVX/AVX2 support" OFF)
option(USE_NEON "Build with NEON support" OFF)
option(BUILD_WEB_TOOLS "Build the web gui tools" ON)
#option(USE_CAFFE_TRIPLET "Build with Caffe Triplet CNN support" OFF)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    message("Compiling for x86 / x86_64 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -march=nocona)

    if(USE_AVX)
        message("AVX/AVX2 support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC OBJ_LOC_USE_AVX_)
        target_compile_options(${PROJECT_NAME} PUBLIC -mavx2 -mfma -march=core-avx2)
    elseif(USE_SSE)
        message("SSE3+ support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC OBJ_LOC_USE_SSE_)
        target_compile_options(${PROJECT_NAME} PUBLIC -msse3)
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(armv7)")
    message("Compiling for ARM v7 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -mcpu=cortex-a53 -mfloat-abi=hard)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC OBJ_LOC_USE_NEON_)
        target_compile_options(${PROJECT_NAME} PUBLIC -mfpu=neon-fp-armv8 -mneon-for-64bits)
    endif(USE_NEON)

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "(aarch64)")
    message("Compiling for aarch64 architectures")

    target_compile_options(${PROJECT_NAME} PUBLIC -march=armv8-a+crypto -mcpu=cortex-a57+crypto)

    if(USE_NEON)
        message("NEON support enabled")
        target_compile_definitions(${PROJECT_NAME} PUBLIC OBJ_LOC_USE_NEON_)
    endif(USE_NEON)

else()
    message(WARNING "Unsupported architecture: using default compiler flags")
endif ()

find_package(OpenMP REQUIRED)
target_compile_options(${PROJECT_NAME} PUBLIC  ${OpenMP_CXX_FLAGS})
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
# The previous line should be replaced in the future with:
# target_link_options(${PROJECT_NAME} PUBLIC  ${OpenMP_EXE_LINKER_FLAGS})

find_package( OpenCV REQUIRED )
# This is to be complinat with OpenCV >= 3.3.1
target_compile_definitions(${PROJECT_NAME} PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )

find_package(Eigen3 REQUIRED)

find_package(OpenGL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_search_module(GLFW REQUIRED glfw3)

set(Python_ADDITIONAL_VERSIONS 3.6)
find_package(PythonLibs 3 REQUIRED)


find_package( Ceres  REQUIRED )
find_package( Boost COMPONENTS program_options filesystem system thread log chrono REQUIRED )

find_package(ZLIB REQUIRED)

include_directories(  ${PROJECT_SOURCE_DIR}/include
                      ${PROJECT_SOURCE_DIR}/${GALAXY_SDK_DIR}/include
                      ${PROJECT_SOURCE_DIR}/${GALAXY_SDK_DIR}/genicam/library/CPP/include
                      ${OpenCV_INCLUDE_DIRS}
                      ${Boost_INCLUDE_DIRS}
                      ${EIGEN3_INCLUDE_DIR}
                      ${CERES_INCLUDE_DIRS}
                      ${ZLIB_INCLUDE_DIRS}
                      ${PYTHON_INCLUDE_DIRS})


target_link_libraries( ${PROJECT_NAME} PUBLIC
        ${OpenMP_CXX_LIBRARIES}
        ${Boost_LIBRARIES}
        ${OpenCV_LIBS}
        ${CERES_LIBRARIES}
        ${ZLIB_LIBRARIES}
        ${PYTHON_LIBRARIES}
        cv_ext
        d2co)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set_target_properties(${PROJECT_NAME} PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
set_target_properties(${PROJECT_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

if ( NOT TARGET cv_ext )
    add_subdirectory(cv_ext )
endif( NOT TARGET cv_ext )

if ( NOT TARGET d2co )
    add_subdirectory(d2co )
endif( NOT TARGET d2co )

add_subdirectory(apps)
