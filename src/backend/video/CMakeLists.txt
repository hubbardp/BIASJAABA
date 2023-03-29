cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(bias_backend_video)
if (POLICY CMP0020) 
    cmake_policy(SET CMP0020 NEW)
endif()

set(
    bias_backend_video_HEADERS
    video_utils.hpp 
    )


set(
    bias_backend_video_SOURCE
    video_utils.cpp 
    )

#qt5_wrap_cpp(bias_backend_video_HEADERS_MOC ${bias_backend_video_HEADERS})
#qt5_wrap_cpp(bias_backend_video_SOURCES_MOC ${bias_backend_video_SOURCE})

add_library(bias_backend_video ${bias_backend_video_SOURCE} ) ##${bias_backend_video_HEADERS})

target_link_libraries(
    bias_backend_video
    Qt5::Core 
    ${bias_ext_link_LIBS} 
    )

