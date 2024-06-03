include_directories(${CMAKE_SOURCE_DIR}/mindspore)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)

include(cmake/op_def.cmake)
include(cmake/ops.cmake)