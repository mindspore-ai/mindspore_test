if(ENABLE_GPU)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR})
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB _GPURES_MANAGER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "*.cc")
    set_property(SOURCE ${_GPURES_MANAGER_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_gpu_obj OBJECT ${_GPURES_MANAGER_LIST})
endif()
