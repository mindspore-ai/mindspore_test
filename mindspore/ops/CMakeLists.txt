include_directories(${CMAKE_SOURCE_DIR}/mindspore)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)

include(cmake/op_def.cmake)
if(ONLY_BUILD_DEVICE_PLUGINS)
    add_library(mindspore_ops INTERFACE) # dummy target to link
else()
    include(cmake/ops.cmake)
endif()