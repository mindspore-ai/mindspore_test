file(GLOB_RECURSE ASCEND_HAL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "hardware/*.cc"
    "profiler/*.cc"
    "special/*.cc")

add_subdirectory(device)
set_property(SOURCE ${ASCEND_HAL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_PROFILER)
add_library(_mindspore_plugin_device_ascend_hal_910b_obj OBJECT ${ASCEND_HAL_SRC_LIST})
