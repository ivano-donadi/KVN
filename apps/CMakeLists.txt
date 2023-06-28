message("Building apps")

add_executable(synergic_localization ${PROJECT_SOURCE_DIR}/apps/synergic_localization.cpp)
target_link_libraries( synergic_localization obj_loc)
set_target_properties( synergic_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(obj_registration ${PROJECT_SOURCE_DIR}/apps/obj_registration.cpp)
target_link_libraries( obj_registration obj_loc)
set_target_properties( obj_registration PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(model_viewer ${PROJECT_SOURCE_DIR}/apps/model_viewer.cpp)
target_link_libraries( model_viewer obj_loc)
set_target_properties( model_viewer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(generate_pvnet_dataset ${PROJECT_SOURCE_DIR}/apps/generate_pvnet_dataset.cpp)
target_link_libraries( generate_pvnet_dataset obj_loc)
set_target_properties( generate_pvnet_dataset PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(generate_d2co_templates ${PROJECT_SOURCE_DIR}/apps/generate_d2co_templates.cpp)
target_link_libraries( generate_d2co_templates obj_loc)
set_target_properties( generate_d2co_templates PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(calib_convert ${PROJECT_SOURCE_DIR}/apps/calib_convert.cpp)
target_link_libraries( calib_convert obj_loc)
set_target_properties( calib_convert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

add_executable(multi_localization ${PROJECT_SOURCE_DIR}/apps/multi_localization.cpp)
target_link_libraries( multi_localization obj_loc)
set_target_properties( multi_localization PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})