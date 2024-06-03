set(OP_DEF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../ops/op_def)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gencode.cmake)

file(GLOB_RECURSE OP_DEF_SRC RELAITVE ${CMAKE_CURRENT_SOURCE_DIR} "${OP_DEF_DIR}/*.cc")
list(REMOVE_ITEM CORE_OPS_LIST "${OP_DEF_DIR}/prim_def.cc")