cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(beh_class)

# beh_class build

set(beh_class_sources
    utils.cpp
    HOGHOF.cpp
    beh_class.cpp
    beh_class_main.cpp)

include_directories(CMAKE_CURRENT_BINARY_DIR)
include_directories(.)
add_executable(beh_class 
               ${beh_class_sources})
target_link_libraries(beh_class 
                      bias_backend_video 
                      heffalump_gpu)
