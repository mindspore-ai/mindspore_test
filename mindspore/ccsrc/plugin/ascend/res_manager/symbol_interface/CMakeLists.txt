if(ENABLE_D OR ENABLE_ACL)
    file(GLOB _SYMBOL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    if(NOT MSVC)
        string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    endif()
    set_property(SOURCE ${_SYMBOL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_ascend_symbol_obj OBJECT ${_SYMBOL_SRC_LIST})
endif()