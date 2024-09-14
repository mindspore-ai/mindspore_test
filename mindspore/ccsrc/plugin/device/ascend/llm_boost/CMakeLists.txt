if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc"
    )

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

file(GLOB_RECURSE MS_LLM_BOOST_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./atb_boost_builder.cc"
        )

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} ${MS_LLM_BOOST_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_LLM_BOOST)

add_library(_mindspore_plugin_device_ascend_llm_boost_910b_obj OBJECT ${MS_LLM_BOOST_910B})
add_subdirectory(atb)
