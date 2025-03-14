include(${OPS_DIR}/cmake/merge_ops.cmake)

file(GLOB_RECURSE PYBOOST_SRC_LIST "*.cc")

# merge customize pyboost kernel files
file(GLOB_RECURSE PYBOOST_MERGE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/customize/*.cc)
list(REMOVE_ITEM PYBOOST_SRC_LIST ${PYBOOST_MERGE_SRC_LIST})

set(ASCEND_PYBOOST_MERGE_DIR ${CMAKE_BINARY_DIR}/merge/mindspore/ops/kernel/ascend/pyboost)
merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/customize ${ASCEND_PYBOOST_MERGE_DIR} ascend_pyboost_merge "")
file(GLOB_RECURSE ASCEND_PYBOOST_MERGE_FILES ${ASCEND_PYBOOST_MERGE_DIR}/*.cc)

add_library(_mindspore_ops_kernel_ascend_pyboost_obj OBJECT ${PYBOOST_SRC_LIST} ${ASCEND_PYBOOST_MERGE_FILES})
