file(GLOB_RECURSE HARDWARE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "collective/*.cc"
  "data_queue/*.cc"
  "device_context/*.cc"
  "event/*.cc"
  "kernel_base/*.cc"
  "memory_manager/*.cc"
  "stream/*.cc")
file(GLOB_RECURSE RES_SRC_LIST "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/device/res_manager/*.cc")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
  string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_subdirectory(gsm)
list(APPEND HARDWARE_SRC_LIST ${RES_SRC_LIST})
set_property(SOURCE ${HARDWARE_SRC_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(mindspore_hardware_abstract SHARED ${HARDWARE_SRC_LIST})
target_link_libraries(mindspore_hardware_abstract PRIVATE mindspore_core mindspore_common
    mindspore_memory_pool mindspore_runtime_pipeline mindspore::pybind11_module)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_hardware_abstract PRIVATE RUNTIME_HARDWARE_DLL)
endif()
set_target_properties(mindspore_hardware_abstract PROPERTIES INSTALL_RPATH $ORIGIN)
