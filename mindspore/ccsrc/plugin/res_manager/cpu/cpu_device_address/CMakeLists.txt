if(ENABLE_CPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE _CPURES_DEVICE_ADDRESS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc")
    set_property(SOURCE ${_CPURES_DEVICE_ADDRESS_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_cpu_cpu_device_address_obj OBJECT ${_CPURES_DEVICE_ADDRESS_LIST})
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_plugin_res_manager_cpu_cpu_device_address_obj PRIVATE CPU_RES_MANAGER_DLL)
    endif()
endif()
