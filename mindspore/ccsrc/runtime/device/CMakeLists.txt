file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/gsm/*.cc"
    "tensors_queue.cc" "device_address_utils.cc"
    "py_execute_utils.cc" "move_to.cc"
)

list(REMOVE_ITEM DEVICE_SRC_LIST "../hardware_abstract/gsm/aio_plugin.cc")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_obj OBJECT ${DEVICE_SRC_LIST})
