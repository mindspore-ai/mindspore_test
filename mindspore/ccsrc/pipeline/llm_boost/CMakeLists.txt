file(GLOB_RECURSE _LLM_BOOST_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "llm_boost_binder.cc"
    "utils.cc"
)

set_property(SOURCE ${_LLM_BOOST_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
                SUBMODULE_ID=mindspore::SubModuleId::SM_LLM_BOOST)

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

add_library(_mindspore_pipeline_llm_boost_obj OBJECT ${_LLM_BOOST_SRC_FILES})
