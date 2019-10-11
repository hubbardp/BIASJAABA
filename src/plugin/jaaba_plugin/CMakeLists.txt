cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(jaaba_plugin)
if (POLICY CMP0020)
    cmake_policy(SET CMP0020 NEW)
endif()

set(
    jaaba_plugin_FORMS 
    jaaba_plugin.ui
    ../../gui/camera_window.ui
    )

set(
    jaaba_plugin_HEADERS 
    jaaba_plugin.hpp
    HOGHOF.hpp
    beh_class.hpp
    process_scores.hpp
    frame_data.hpp
    shape_data.hpp
    )

set(
    jaaba_plugin_SOURCES 
    jaaba_plugin.cpp
    HOGHOF.cpp
    beh_class.cpp
    process_scores.cpp
    )

qt5_wrap_ui(jaaba_plugin_FORMS_HEADERS ${jaaba_plugin_FORMS})
qt5_wrap_cpp(jaaba_plugin_HEADERS_MOC ${jaaba_plugin_HEADERS})

add_library(
    jaaba_plugin 
    ${jaaba_plugin_HEADERS_MOC}
    ${jaaba_plugin_FORMS_HEADERS}
    ${jaaba_plugin_SOURCES} 
    )

add_dependencies(jaaba_plugin ${jaaba_plugin_FORMS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)
message(STATUS ${CMAKE_CURRENT_BINARY_DIR}) 
target_link_libraries(jaaba_plugin ${QT_LIBRARIES} bias_plugin bias_utility bias_camera_facade qcustomplot heffalump_gpu)

qt5_use_modules(jaaba_plugin Core Widgets Gui PrintSupport SerialPort)


