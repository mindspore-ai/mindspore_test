if(ENABLE_CPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE _CPURES_MEM_MANAGER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc")
    set_property(SOURCE ${_CPURES_MEM_MANAGER_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_cpu_cpu_mem_manager_obj OBJECT ${_CPURES_MEM_MANAGER_LIST})
endif()
