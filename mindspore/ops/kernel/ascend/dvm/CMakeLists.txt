file(GLOB_RECURSE DVM_SRC ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(_mindspore_ops_kernel_ascend_dvm_obj OBJECT ${DVM_SRC})
