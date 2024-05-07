cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(flytrack_plugin)
if (POLICY CMP0020)
    cmake_policy(SET CMP0020 NEW)
endif()

set(
    flytrack_plugin_HEADERS 
    flytrack_plugin.hpp
    ../../gui/camera_window.hpp
    )

set(
    flytrack_plugin_SOURCES 
    flytrack_plugin.cpp
    )

qt5_wrap_ui(ui_headers ../../gui/camera_window.ui)

qt5_wrap_cpp(flytrack_plugin_HEADERS_MOC ${flytrack_plugin_HEADERS})

add_library(
    flytrack_plugin 
    ${flytrack_plugin_HEADERS_MOC}
    ${flytrack_plugin_SOURCES} 
    ${ui_headers}
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)
target_link_libraries(flytrack_plugin ${QT_LIBRARIES} ${OpenCV_LIBRARIES})

qt5_use_modules(flytrack_plugin Core Widgets Gui)

