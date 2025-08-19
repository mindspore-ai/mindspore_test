
if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
  string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)

file(GLOB_RECURSE RUNTIME_CORE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

set_property(SOURCE ${RUNTIME_CORE_SRC_LIST}
  PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)

add_library(_mindspore_runtime_core_obj OBJECT ${RUNTIME_CORE_SRC_LIST})

target_link_libraries(_mindspore_runtime_core_obj PRIVATE mindspore_backend_common)
