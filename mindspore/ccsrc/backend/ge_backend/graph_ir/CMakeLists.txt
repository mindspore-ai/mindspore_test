if(ENABLE_D OR ENABLE_ACL)
    file(GLOB_RECURSE _TRANSFORM_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_TRANSFORM_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
      SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)

    # mindspore_graph_ir is used by GE and lite.
    if(TARGET mindspore_ascend_res_manager)
        add_library(mindspore_graph_ir SHARED ${_TRANSFORM_SRC_LIST})
        target_link_libraries(mindspore_graph_ir PRIVATE mindspore_ascend_res_manager)
    else()
        file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../version.txt" VERSION)
        add_definitions(-DVERSION="${VERSION}")
        list(REMOVE_ITEM _TRANSFORM_SRC_LIST "callbacks_ge.cc")
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../plugin/ascend/res_manager/op_adapter/
            _mindspore_ascend_op_adapter_obj)
        add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../../plugin/ascend/res_manager/symbol_interface
            _mindspore_ascend_symbol_obj)
        list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/config_manager.cc")
        list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/common.cc")

        if(NOT WIN32)
            list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/anfalgo.cc")
            list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/utils.cc")
            list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/parallel_context.cc")
        endif()
        list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/convert_utils.cc")

        list(APPEND _TRANSFORM_SRC_LIST
                "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hardware_abstract/kernel_base/kernel_info.cc")
        list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/comm_manager.cc")
        list(APPEND _TRANSFORM_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/../../../utils/compile_cache_context.cc")

        list(APPEND _TRANSFORM_SRC_LIST
            "${CMAKE_CURRENT_SOURCE_DIR}/../../../runtime/hardware_abstract/kernel_base/kernel_build_info.cc")
        add_library(mindspore_graph_ir SHARED ${_TRANSFORM_SRC_LIST} $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>
            $<TARGET_OBJECTS:_mindspore_ascend_op_adapter_obj>)
    endif()

    target_link_libraries(mindspore_graph_ir PRIVATE mindspore_core mindspore_ops)
    find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    target_link_libraries(mindspore_graph_ir PRIVATE ${GE_RUNNER} ${GRAPH} ${ACL})
    set_target_properties(mindspore_graph_ir PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/../..)
endif()
