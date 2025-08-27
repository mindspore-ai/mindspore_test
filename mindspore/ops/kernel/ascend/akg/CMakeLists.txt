file(GLOB_RECURSE AKG_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(_mindspore_ops_kernel_ascend_akg_obj OBJECT ${AKG_SRC})
