cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(opencv_test)
if (POLICY CMP0020) 
    cmake_policy(SET CMP0020 NEW)
endif()
if (POLICY CMP0043) 
    cmake_policy(SET CMP0043 NEW)
endif()

# opencv_test build

set(
    opencv_test_HEADERS 
	opencv_videowrite_test.hpp
   )

set(
    opencv_test_sources
	opencv_videowrite_test.cpp
    )


include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)


add_executable(
               opencv_test
               ${opencv_test_HEADERS}
               ${opencv_test_sources}
              )
     
target_link_libraries(
                      opencv_test
                      ${OpenCV_LIBS}
                     )