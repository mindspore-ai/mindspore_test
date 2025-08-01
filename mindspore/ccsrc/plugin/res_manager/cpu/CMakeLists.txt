if(ENABLE_CPU)
    file(GLOB_RECURSE _CPU_RESMANAGER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_CPU_RESMANAGER_SRC_LIST}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_res_manager_cpu_obj OBJECT ${_CPU_RESMANAGER_SRC_LIST})
endif()
