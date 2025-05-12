if(ENABLE_CPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB _CPURES_MANAGER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc")
    set_property(SOURCE ${_CPURES_MANAGER_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_cpu_obj OBJECT ${_CPURES_MANAGER_LIST})
endif()
