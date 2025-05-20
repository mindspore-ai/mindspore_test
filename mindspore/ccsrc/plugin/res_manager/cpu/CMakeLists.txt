if(ENABLE_CPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB _CPURES_MANAGER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc")
    set_property(SOURCE ${_CPURES_MANAGER_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_cpu_obj OBJECT ${_CPURES_MANAGER_LIST})
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_plugin_res_manager_cpu_obj PRIVATE CPU_RES_MANAGER_DLL)
    endif()
    add_subdirectory(cpu_device_address)
    add_subdirectory(cpu_mem_manager)

    list(APPEND CPU_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_manager_cpu_obj>)
    list(APPEND CPU_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_manager_cpu_cpu_device_address_obj>)
    list(APPEND CPU_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_manager_cpu_cpu_mem_manager_obj>)

    add_library(mindspore_cpu_res_manager SHARED ${CPU_RES_MANAGER_OBJECTS_SRC})
    target_link_libraries(mindspore_cpu_res_manager PRIVATE mindspore_core mindspore_common mindspore_dump
        mindspore_profiler mindspore_ops_kernel_common mindspore_res_manager mindspore_memory_pool securec)
    set_target_properties(mindspore_cpu_res_manager PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/../..)
endif()
