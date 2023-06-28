project( obj_loc )
message("** Project ${PROJECT_NAME} **")

cmake_minimum_required(VERSION 3.0)

option(STANDALONE_BUILD "Build as a standalone package, all testing and debugging tools, and not as hyro module." OFF)

if (STANDALONE_BUILD)
    include(standalone_build/CMakeLists.txt)
else()
    include(hyro_build/CMakeLists.txt)
endif ()