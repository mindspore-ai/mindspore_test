include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)

file(GLOB_RECURSE _GRAPH_COMPILER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_GRAPH_COMPILER_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_VM)
add_library(_mindspore_backend_graph_compiler_obj OBJECT ${_GRAPH_COMPILER_SRC_LIST})

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()