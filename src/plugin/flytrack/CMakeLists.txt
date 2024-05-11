cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(flytrack_plugin)
if (POLICY CMP0020)
    cmake_policy(SET CMP0020 NEW)
endif()

set(
    flytrack_plugin_FORMS 
    flytrack_plugin.ui
    ../../gui/camera_window.ui
    )


set(
    flytrack_plugin_HEADERS
    flytrack_config.hpp
    flytrack_plugin.hpp
    )

set(
    flytrack_plugin_SOURCES
    flytrack_config.cpp
    flytrack_plugin.cpp
    )

qt5_wrap_ui(flytrack_plugin_FORMS_HEADERS ${flytrack_plugin_FORMS}) 

qt5_wrap_cpp(flytrack_plugin_HEADERS_MOC ${flytrack_plugin_HEADERS})

add_library(
    flytrack_plugin 
    ${flytrack_plugin_HEADERS_MOC}
    ${flytrack_plugin_FORMS_HEADERS}
    ${flytrack_plugin_SOURCES} 
    )

add_dependencies(flytrack_plugin ${flytrack_plugin_FORMS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)
target_link_libraries(flytrack_plugin ${QT_LIBRARIES} bias_plugin bias_utility bias_camera_facade qcustomplot)
#target_link_libraries(flytrack_plugin ${QT_LIBRARIES} ${OpenCV_LIBRARIES})

qt5_use_modules(flytrack_plugin Core Widgets Gui)

