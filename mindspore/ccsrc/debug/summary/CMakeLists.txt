file(GLOB_RECURSE SUMMARY_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
add_library(_mindspore_debug_summary_obj OBJECT ${SUMMARY_SRC_LIST})
add_dependencies(_mindspore_debug_summary_obj proto_input)