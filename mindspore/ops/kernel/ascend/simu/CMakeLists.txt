file(GLOB_RECURSE _SIMU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_ops_kernel_ascend_simu_obj OBJECT ${_SIMU_SRC_LIST})
