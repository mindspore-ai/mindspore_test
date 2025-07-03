file(GLOB_RECURSE RUNTIME_CONF_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc")

add_library(_mindspore_common_runtime_conf_obj OBJECT ${RUNTIME_CONF_SRC_LIST})
