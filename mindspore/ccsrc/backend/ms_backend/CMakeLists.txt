file(GLOB_RECURSE _BACKEND_MSBACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
        )
if(WIN32)
        list(APPEND _BACKEND_MSBACKEND_SRC_LIST
        "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/cpu/cpu_device_address_maker.cc")
endif()
set_property(SOURCE ${_BACKEND_MSBACKEND_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(_mindspore_backend_ms_backend_obj OBJECT ${_BACKEND_MSBACKEND_SRC_LIST})