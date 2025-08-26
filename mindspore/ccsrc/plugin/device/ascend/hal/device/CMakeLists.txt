if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE MS_DEVICE_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${MS_DEVICE_910B} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

add_library(_mindspore_plugin_device_ascend_hal_device_910b_obj OBJECT ${MS_DEVICE_910B})