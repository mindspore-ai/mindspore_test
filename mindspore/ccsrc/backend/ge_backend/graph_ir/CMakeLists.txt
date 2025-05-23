if(ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    if(BUILD_LITE)
        list(REMOVE_ITEM _TRANSFORM_SRC_LIST "callbacks_ge.cc")
    endif()
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)

      # mindspore_graph_ir is used by GE and lite.
    if(BUILD_LITE)
        add_subdirectory(${TOP_DIR}/mindspore/ccsrc/plugin/res_manager/ascend/op_adapter/
            _mindspore_ascend_op_adapter_obj)
        add_subdirectory(${TOP_DIR}/mindspore/ccsrc/plugin/res_manager/ascend/symbol_interface
            _mindspore_ascend_symbol_obj)
        list(APPEND _TRANSFORM_SRC_LIST "${TOP_DIR}/mindspore/ccsrc/utils/config_manager.cc")
        add_library(mindspore_graph_ir SHARED ${_TRANSFORM_SRC_LIST} $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>
            $<TARGET_OBJECTS:_mindspore_ascend_op_adapter_obj>)
    else()
        add_library(mindspore_graph_ir SHARED ${_TRANSFORM_SRC_LIST})
        target_link_libraries(mindspore_graph_ir PRIVATE mindspore_ascend_res_manager)
    endif()

    target_link_libraries(mindspore_graph_ir PRIVATE mindspore_core mindspore_ops)
    find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_graph_ir PRIVATE ${GE_RUNNER} ${GRAPH} ${ACL})
    set_target_properties(mindspore_graph_ir PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/../..)
endif()
