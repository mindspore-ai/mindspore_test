file(GLOB_RECURSE _COMMON_IR_DUMP_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
file(STRINGS "${TOP_DIR}/version.txt" VERSION)
add_definitions(-DVERSION=\"${VERSION}\")
set_property(SOURCE ${_COMMON_IR_DUMP_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
add_library(_mindspore_common_ir_dump_obj OBJECT ${_COMMON_IR_DUMP_SRC_LIST})
