file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "common/*.cc" "gsm/*.cc"
    "executor/dynamic_kernel.cc" "executor/executor_callback.cc"
    "kernel_runtime.cc" "kernel_runtime_manager.cc" "memory_scheduler.cc"
    "memory_offload_strategy.cc" "launch_kernel.cc" "launch_mul.cc"
    "context_extends.cc" "stream_synchronizer.cc" "tensors_queue.cc"
    "common_somas_allocator.cc" "device_address_utils.cc"
    "py_execute_utils.cc" "move_to.cc"
)

list(REMOVE_ITEM DEVICE_SRC_LIST "gsm/aio_plugin.cc")
add_subdirectory(gsm)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_device_obj OBJECT ${DEVICE_SRC_LIST})
