if(ENABLE_GPU)
  file(GLOB _GPU_RES_MANAGER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  set_property(SOURCE ${_GPU_RES_MANAGER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
  add_library(_mindspore_gpu_res_manager_obj OBJECT ${_GPU_RES_MANAGER_SRC_LIST})

  add_subdirectory(device_context_conf)
  add_subdirectory(device)

  list(APPEND GPU_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_mananger_gpu_device_context_conf_obj>)
  list(APPEND GPU_RES_MANAGER_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_manager_gpu_device_obj>)

  add_library(mindspore_gpu_res_manager SHARED ${GPU_RES_MANAGER_OBJECTS_SRC})
  target_link_libraries(mindspore_gpu_res_manager PRIVATE mindspore_core mindspore_common pybind11::module
    mindspore_res_manager)
  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
    target_link_libraries(mindspore_gpu_res_manager PRIVATE gpu_hash_table)
  endif()
  set_target_properties(mindspore_gpu_res_manager PROPERTIES INSTALL_RPATH $ORIGIN)
endif()
