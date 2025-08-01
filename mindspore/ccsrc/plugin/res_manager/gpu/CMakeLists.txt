if(ENABLE_GPU)
  file(GLOB _GPU_RES_MANAGER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
  set_property(SOURCE ${_GPU_RES_MANAGER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
  add_library(_mindspore_gpu_res_manager_obj OBJECT ${_GPU_RES_MANAGER_SRC_LIST})

  add_subdirectory(device_context_conf)
  add_subdirectory(device)
endif()
