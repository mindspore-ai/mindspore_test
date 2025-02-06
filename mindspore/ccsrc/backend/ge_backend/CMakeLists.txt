if(ENABLE_D OR ENABLE_ACL)
    include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)

    file(GLOB _GE_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_runtime_obj>)


    set_property(SOURCE ${_GE_BACKEND_SRC_LIST}
          PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
    add_library(mindspore_ge_backend SHARED ${_GE_BACKEND_SRC_LIST})

    target_link_libraries(mindspore_ge_backend PRIVATE mindspore_core mindspore_ops
            mindspore_common mindspore_backend_manager mindspore_backend mindspore_ascend proto_input)
    target_link_libraries(mindspore_ge_backend PRIVATE securec)


    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(mindspore_ge_backend PROPERTIES MACOSX_RPATH ON)
        set_target_properties(mindspore_ge_backend PROPERTIES INSTALL_RPATH @loader_path:@loader_path/plugin)
    else()
        set_target_properties(mindspore_ge_backend PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin)
    endif()

    add_subdirectory(pass)
    add_subdirectory(executor)
    add_subdirectory(runtime)
endif()
