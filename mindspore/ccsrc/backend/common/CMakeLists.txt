file(GLOB_RECURSE BACKEND_COMMON_SUB_OBJECTS_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

set_property(SOURCE ${BACKEND_COMMON_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_SESSION)
add_library(mindspore_backend_common SHARED ${BACKEND_COMMON_SUB_OBJECTS_SRC})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_backend_common PRIVATE BACKEND_COMMON_DLL)
endif()

target_link_libraries(mindspore_backend_common PRIVATE mindspore_core mindspore_ops mindspore_common
    proto_input mindspore_memory_pool mindspore_hardware_abstract mindspore_profiler
    mindspore_runtime_pipeline mindspore_runtime_utils)
