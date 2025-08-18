file(GLOB_RECURSE _JIT_TRACE_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "trace_recorder.cc"
)

file(GLOB _JIT_TRACE_SRC_FILES "*.cc")
set_property(SOURCE ${_JIT_TRACE_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_TRACE)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_library(_mindspore_frontend_jit_trace_obj OBJECT ${_JIT_TRACE_SRC_FILES})
