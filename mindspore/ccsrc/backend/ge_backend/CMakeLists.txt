if(ENABLE_D OR ENABLE_ACL)
    include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)

    file(GLOB _GE_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_runtime_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_dump_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_utils_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_pass_obj>)
    list(APPEND _GE_BACKEND_SRC_LIST $<TARGET_OBJECTS:_mindspore_ge_executor_obj>)


    set_property(SOURCE ${_GE_BACKEND_SRC_LIST}
          PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
    add_library(mindspore_ge_backend SHARED ${_GE_BACKEND_SRC_LIST})

    target_link_libraries(mindspore_ge_backend PRIVATE mindspore_core mindspore_ops mindspore_ascend_res_manager
            mindspore_common mindspore_backend_manager mindspore_backend_common proto_input mindspore_runtime_pipeline
            mindspore_memory_pool)
    target_link_libraries(mindspore_ge_backend PRIVATE securec d_collective)

    set_target_properties(mindspore_ge_backend PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin:$ORIGIN/plugin/ascend)

    add_subdirectory(pass)
    add_subdirectory(executor)
    add_subdirectory(runtime)
    add_subdirectory(dump)
    add_subdirectory(utils)
    add_subdirectory(graph_ir)

    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_ge_backend PRIVATE ${GE_RUNNER} ${GRAPH})
    target_link_libraries(mindspore_ge_backend PRIVATE mindspore::event mindspore::event_pthreads
        mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()
