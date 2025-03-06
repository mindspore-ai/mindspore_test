include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

file(GLOB_RECURSE MS_LLM_BOOST_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./boost_model_atb.cc"
        "./workspace.cc"
        "./buffer_device.cc"
        )

set_property(SOURCE ${MS_LLM_BOOST_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_LLM_BOOST)

add_library(ms_atb_boost SHARED ${MS_LLM_BOOST_910B})

target_link_libraries(ms_atb_boost PRIVATE mindspore_backend mindspore_core mindspore_common)
target_link_libraries(ms_atb_boost PRIVATE proto_input mindspore::protobuf)
target_link_libraries(ms_atb_boost PRIVATE securec d_collective)
target_link_libraries(ms_atb_boost PRIVATE mindspore::ssl mindspore::crypto)
