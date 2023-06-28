message("Building examples")

add_executable(test_conversions ${PROJECT_SOURCE_DIR}/examples/test_conversions.cpp)
target_link_libraries( test_conversions cv_ext)  
set_target_properties( test_conversions PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(test_skeleton ${PROJECT_SOURCE_DIR}/examples/test_skeleton.cpp)
target_link_libraries( test_skeleton cv_ext)
set_target_properties( test_skeleton PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(test_projector ${PROJECT_SOURCE_DIR}/examples/test_projector.cpp)
target_link_libraries( test_projector cv_ext)                       
set_target_properties( test_projector PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(image_gradient_example ${PROJECT_SOURCE_DIR}/examples/image_gradient_example.cpp)
target_link_libraries( image_gradient_example cv_ext)
set_target_properties( image_gradient_example PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(test_hdr ${PROJECT_SOURCE_DIR}/examples/test_hdr.cpp)
target_link_libraries( test_hdr cv_ext)
set_target_properties( test_hdr PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(test_shapes ${PROJECT_SOURCE_DIR}/examples/test_shapes.cpp)
target_link_libraries( test_shapes cv_ext)
set_target_properties( test_shapes PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(test_image_inspection ${PROJECT_SOURCE_DIR}/examples/test_image_inspection.cpp)
target_link_libraries( test_image_inspection cv_ext)
set_target_properties( test_image_inspection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(cam_calib ${PROJECT_SOURCE_DIR}/examples/cam_calib.cpp ${PROJECT_SOURCE_DIR}/examples/apps_utils.cpp )
target_link_libraries( cam_calib cv_ext)
set_target_properties( cam_calib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )

add_executable(stereo_cam_calib ${PROJECT_SOURCE_DIR}/examples/stereo_cam_calib.cpp 
                                ${PROJECT_SOURCE_DIR}/examples/apps_utils.cpp )
target_link_libraries( stereo_cam_calib cv_ext)
set_target_properties( stereo_cam_calib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )

add_executable(stereo_rectify ${PROJECT_SOURCE_DIR}/examples/stereo_rectify.cpp ${PROJECT_SOURCE_DIR}/examples/apps_utils.cpp )
target_link_libraries( stereo_rectify cv_ext)
set_target_properties( stereo_rectify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )

add_executable(multistereo_cam_calib ${PROJECT_SOURCE_DIR}/examples/multistereo_cam_calib.cpp
                                     ${PROJECT_SOURCE_DIR}/examples/apps_utils.cpp)
target_link_libraries( multistereo_cam_calib cv_ext)
set_target_properties( multistereo_cam_calib PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY} )
