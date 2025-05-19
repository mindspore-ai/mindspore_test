if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

file(GLOB_RECURSE MS_HARDWARE_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "ascend_deprecated_interface.cc"
        "ascend_device_context.cc"
        "ge_summary.cc"
        "ascend_device_res_manager.cc"
        "ge_device_res_manager.cc"
        "ge_memory_allocator.cc"
        "ge_memory_manager.cc"
        "ge_graph_executor.cc"
        "ascend_kernel_executor.cc"
        "ge_utils.cc"
        "ascend_graph_optimization.cc"
        "acl_somas.cc"
        "acl_stream_assign.cc"
        "stress_detect.cc"
        "ascend_device_address_maker.cc"
        )

set_property(SOURCE ${MS_HARDWARE_910B} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

add_library(_mindspore_plugin_device_ascend_hal_hardware_910b_obj OBJECT ${MS_HARDWARE_910B})
add_library(gpto OBJECT gpto.cc)
target_include_directories(gpto PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mindspore_frontend PRIVATE gpto)