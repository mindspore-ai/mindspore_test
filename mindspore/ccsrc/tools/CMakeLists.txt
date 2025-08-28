include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/tools/)
include_directories(${CMAKE_BINARY_DIR})
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

set(_DEBUG_SRC_LIST)

if(ENABLE_DUMP_IR)
    file(GLOB_RECURSE _RDR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "rdr/*.cc")
endif()

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_DEBUGGER)
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debug_services.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/debugger_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_stat_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/utils.cc"
        )
else()
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/debugger/proto_exporter_stub.cc"
        )
endif()

list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/execute_order_tracker/execute_order_tracker.cc"
    )

list(APPEND _DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/common/csv_writer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/summary/summary.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/summary/callbacks.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/summary/event_writer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_control.cc"
    )
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/e2e_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_statistic.cc"
        )
    file(GLOB_RECURSE _DEVICE_STATISTIC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/device_statistic/*.cc")
endif()

set_property(SOURCE ${_DEBUG_SRC_LIST} ${_RDR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEBUG)
list(APPEND _DEBUG_SRC_LIST ${_RDR_SRC_LIST})
file(GLOB_RECURSE _ERROR_HANDLER_LIST "${CMAKE_CURRENT_SOURCE_DIR}/error_handler/*.cc")
list(APPEND _DEBUG_SRC_LIST ${_ERROR_HANDLER_LIST})
list(LENGTH _DEBUG_SRC_LIST debug_files_size)
list(APPEND _DEBUG_SRC_LIST ${_DEVICE_STATISTIC_LIST})
file(GLOB_RECURSE _CHECKSUM_LIST "${CMAKE_CURRENT_SOURCE_DIR}/checksum/*.cc")
list(APPEND _DEBUG_SRC_LIST ${_CHECKSUM_LIST})
if(${debug_files_size} GREATER 0)
  add_library(_mindspore_tools_obj OBJECT ${_DEBUG_SRC_LIST})
  add_dependencies(_mindspore_tools_obj proto_input)
endif()


set(DEBUG_SUB_COMP
    dump
    silent_detect
)

foreach(_comp ${DEBUG_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND DEBUG_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE DEBUG_DLL)
        endif()
    endif()
endforeach()

add_library(mindspore_dump SHARED ${DEBUG_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_dump PRIVATE mindspore_core proto_input)
set_target_properties(mindspore_dump PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_dump PRIVATE mindspore::pybind11_module)
endif()
