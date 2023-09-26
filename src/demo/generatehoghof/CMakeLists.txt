cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(beh_class)
if (POLICY CMP0020) 
    cmake_policy(SET CMP0020 NEW)
endif()
if (POLICY CMP0043) 
    cmake_policy(SET CMP0043 NEW)
endif()

# generate_hoghof build

set(
    generate_hoghof_sources
    generate_hoghof.cpp
    )

include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(.)


add_executable(
               generate_hoghof
               ${generate_hoghof_sources}
              )
     
target_link_libraries(
                      generate_hoghof
					  bias_utility
                      heffalump_gpu 
                      bias_backend_video   
					  jaaba_plugin
                     )

##qt5_use_modules(beh_class Core Gui Widgets)