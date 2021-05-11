# - Try to find NIDAQmx
# 
# Once done this will define
#
#  NIDAQmx_FOUND         - System has FlyCapture2
#  NIDAQmx_INCLUDE_DIRS  - The FlyCapture2 include directories
#  NIDAQmx_LIBRARIES    - The libraries needed to use FlyCapture2
#
# ------------------------------------------------------------------------------

if (WIN32)
    set(typical_nidaqmx_dir "C:/Program Files (x86)/National Instruments/Shared/ExternalCompilerSupport/C")
    set(typical_nidaqmx_lib_dir "${typical_nidaqmx_dir}/lib64/msvc/")
    set(typical_nidaqmx_inc_dir "${typical_nidaqmx_dir}/include/")
endif()

message(STATUS "${typical_nidaqmx_dir}")

message(STATUS "finding include dir")
find_path(
    NIDAQmx_INCLUDE_DIR 
    "NIDAQmx.h"
    HINTS ${typical_nidaqmx_inc_dir}
    )
message(STATUS "NIDAQmx_INCLUDE_DIR: " ${NIDAQmx_INCLUDE_DIR})

if(WIN32)
    message(STATUS "finding library")
    find_library(
        NIDAQmx_LIBRARY 
        NAMES "NIDAQmx.lib"
        HINTS ${typical_nidaqmx_lib_dir} 
        )
endif()


set(NIDAQmx_LIBRARIES ${NIDAQmx_LIBRARY} )
set(NIDAQmx_INCLUDE_DIRS ${NIDAQmx_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set NIDAQmx_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
    NIDAQmx DEFAULT_MSG
    NIDAQmx_LIBRARY 
    NIDAQmx_INCLUDE_DIR
    )

mark_as_advanced(NIDAQmx_INCLUDE_DIR NIDAQmx_LIBRARY )

