hyro_init_package(ar_object_localization VERSION 0.0.0)

hyro_find_package(hyro_utils REQUIRED)
hyro_find_package(hyro_msgs REQUIRED)
hyro_find_package(hyro_tools_media REQUIRED)
hyro_find_package(ar_object_detection REQUIRED)

find_package(PCL REQUIRED )
find_package(OpenMP REQUIRED)

add_definitions( ${PCL_DEFINITIONS}
                 -DTW_NO_LIB_PRAGMA
                 -DTW_NO_DIRECT3D
                 -DGLM_FORCE_RADIANS)


hyro_add_library(
  ${PROJECT_NAME}_msgs
  HEADERS
    hyro/include/msgs/ar_object_localization/ObjectLocalizationItem.h
    hyro/include/msgs/ar_object_localization/ObjectLocalizationItem.proto.h
  DEPENDS_ON
    hyro_msgs
    protobuf::libprotobuf
)

hyro_compile_messages(
  ${PROJECT_NAME}_msgs SHARED
  SOURCES
    hyro/msgs/hyro/msgs/ar_object_localization/ObjectLocalizationItem.proto
  DEPENDS_ON
    hyro_msgs
)

hyro_add_library(
  ${PROJECT_NAME}_core
  HEADERS 
  SOURCES 
    src/TMObjectLocalization.cpp
  INCLUDE_DIRS
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/${GALAXY_SDK_DIR}/include
    ${PROJECT_SOURCE_DIR}/${GALAXY_SDK_DIR}/genicam/library/CPP/include
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${PCL_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/d2co/include
    ${PROJECT_SOURCE_DIR}/cv_ext/include

  DEPENDS_ON
    cv_ext
    d2co
    ${OpenMP_CXX_LIBRARIES}
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
    ${PCL_LIBRARIES}
    ${CERES_LIBRARIES}
)

target_compile_options(${PROJECT_NAME}_core PUBLIC -std=c++11 -Wall -Wpedantic -Wno-narrowing -Wno-deprecated 
                                              -pthread -O3 -ffast-math -fPIC)
target_compile_options(${PROJECT_NAME}_core PUBLIC -msse3)
target_compile_options(${PROJECT_NAME}_core PUBLIC  ${OpenMP_CXX_FLAGS})
target_compile_definitions(${PROJECT_NAME}_core PUBLIC OPENCV_TRAITS_ENABLE_DEPRECATED )


hyro_add_library(
  ${PROJECT_NAME}_components
  HEADERS
        hyro/include/ar_object_localization_component.h
  SOURCES
        hyro/components/ar_object_localization_component.cpp
  INCLUDE_DIRS
    include/
    #${OpenCV_INCLUDE_DIRS}

  DEPENDS_ON
    hyro_msgs
    ${PROJECT_NAME}_msgs
    hyro_utils
    hyro_tools_media
    ar_object_detection_msgs
    ${PROJECT_NAME}_core
)



hyro_add_executable(${PROJECT_NAME}
  SOURCES hyro/apps/ar_object_localization_main.cpp
  HEADERS

  DEPENDS_ON 
    ${PROJECT_NAME}_components
)

hyro_add_executable(ar_tm_localization
  SOURCES 
    ${PROJECT_SOURCE_DIR}/apps/tm_localization.cpp 
    ${PROJECT_SOURCE_DIR}/apps/apps_utils.cpp
  HEADERS

  DEPENDS_ON 
    ${PROJECT_NAME}_core
)

hyro_add_executable(ar_gen_templates
  SOURCES 
    ${PROJECT_SOURCE_DIR}/apps/generate_templates.cpp 
    ${PROJECT_SOURCE_DIR}/apps/apps_utils.cpp
  HEADERS

  DEPENDS_ON 
    ${PROJECT_NAME}_core
)

hyro_add_executable(ar_obj_registration 
  SOURCES 
    ${PROJECT_SOURCE_DIR}/apps/obj_registration.cpp 
    ${PROJECT_SOURCE_DIR}/apps/apps_utils.cpp
  HEADERS

  DEPENDS_ON 
    ${PROJECT_NAME}_core
)

# hyro_install_package(${PROJECT_NAME}
#   EXES
#     ${PROJECT_NAME}
#     ar_tm_localization
#     ar_gen_templates
#     ar_obj_registration
#   LIBS
#     ${PROJECT_NAME}_msgs
# #     ${PROJECT_NAME}_core
# #     ${PROJECT_NAME}_components
# )

if ( NOT TARGET cv_ext )
    add_subdirectory(cv_ext )
endif( NOT TARGET cv_ext )

if ( NOT TARGET d2co )
    add_subdirectory(d2co )
endif( NOT TARGET d2co )
