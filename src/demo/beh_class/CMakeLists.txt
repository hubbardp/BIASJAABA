cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(beh_class)

# beh_class build

set(
    beh_class_HEADERS 
    utils.hpp
    HOGHOF.hpp
    beh_class.hpp
   )

set(
    beh_class_sources
    beh_class_main.cpp
    utils.cpp
    HOGHOF.cpp
    beh_class.cpp
    )

qt5_wrap_cpp(beh_class_HEADERS_MOC ${JAABA_plugin_HEADERS})

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)


add_executable(
               beh_class
               ${beh_class_HEADERS_MOC}
               ${beh_class_sources}
              )
     
target_link_libraries(
                      beh_class
                      heffalump_gpu 
                      ${QT_LIBRARIES}
                      bias_backend_video 
                      bias_utility  
                     )

qt5_use_modules(beh_class Core Gui Widgets)

