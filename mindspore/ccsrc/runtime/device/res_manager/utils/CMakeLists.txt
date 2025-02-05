include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

file(GLOB RES_UTILS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc")

set_property(SOURCE ${RES_UTILS_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_res_manager_utils_obj OBJECT ${RES_UTILS_LIST})
