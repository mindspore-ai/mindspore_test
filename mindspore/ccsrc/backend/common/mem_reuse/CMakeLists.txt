file(GLOB_RECURSE _PREACTIVATE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -finline-functions -funswitch-loops \
          -fgcse-after-reload -finline-small-functions -fpredictive-commoning")
    endif()
endif()

set_property(SOURCE ${_PREACTIVATE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_PRE_ACT)
add_library(_mindspore_backend_common_mem_reuse_obj OBJECT ${_PREACTIVATE_SRC_LIST})
