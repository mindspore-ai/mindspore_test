file(GLOB_RECURSE HARDWARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "device_context_manager.cc" "device_context.cc")
list(APPEND HARDWARE_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../graph_scheduler/device_tensor_store.cc")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${HARDWARE_SRC_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_runtime_hardware_obj OBJECT ${HARDWARE_SRC_LIST})
