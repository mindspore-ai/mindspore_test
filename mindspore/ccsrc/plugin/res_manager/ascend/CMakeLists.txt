if(ENABLE_D OR ENABLE_ACL)
  file(GLOB _ASCEND_RES_MANAGER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  set_property(SOURCE ${_ASCEND_RES_MANAGER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
  add_library(_mindspore_ascend_res_manager_obj OBJECT ${_ASCEND_RES_MANAGER_SRC_LIST})

  add_subdirectory(op_adapter)
  add_subdirectory(symbol_interface)
  add_subdirectory(device_context_conf)
  add_subdirectory(stream_manager)
  add_subdirectory(hal_manager)
  add_subdirectory(mbuf_manager)

  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_op_adapter_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_symbol_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_context_conf_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_stream_manager_obj>)
  list(APPEND ASCEND_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_ascend_hal_manager_obj>)

  add_library(mindspore_ascend_res_manager SHARED ${ASCEND_RES_MANAGER_OBJECTS_SRC})
  target_link_libraries(mindspore_ascend_res_manager PRIVATE mindspore_core mindspore_common pybind11::module
    proto_input)
  # if(ENABLE_NATIVE_PROTOBUF)
  #     target_link_libraries(mindspore_ascend_res_manager PRIVATE ${PROTOBUF_LIB})
  # elseif(NOT TARGET_AOS_ARM AND NOT ANDROID_NDK)
  #     target_link_libraries(mindspore_ascend_res_manager PRIVATE mindspore::protobuf)
  # endif()
  set_target_properties(mindspore_ascend_res_manager PROPERTIES INSTALL_RPATH $ORIGIN)
endif()
