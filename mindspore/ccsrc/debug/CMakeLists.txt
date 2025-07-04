include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/debug/)
include_directories(${CMAKE_BINARY_DIR})
file(STRINGS "${CMAKE_SOURCE_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")

add_subdirectory(dump)

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
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/tensor_summary.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debug_services.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/debugger_utils.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/tensor_stat_dump.cc"
        "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc"
        )
else()
    list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/debugger/proto_exporter_stub.cc"
        )
endif()

list(APPEND _DEBUG_SRC_LIST
        "${CMAKE_CURRENT_SOURCE_DIR}/execute_order_tracker/execute_order_tracker.cc"
    )

list(APPEND _DEBUG_SRC_LIST
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/cpu_e2e_dump.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_json_parser.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/data_dump/dump_utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/utils.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/csv_writer.cc"
    "${CMAKE_CURRENT_SOURCE_DIR}/summary/summary.cc"
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
file(GLOB_RECURSE _TFT_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/tft_adapter/*.cc")
list(APPEND _DEBUG_SRC_LIST ${_TFT_SRC_LIST})
list(LENGTH _DEBUG_SRC_LIST debug_files_size)
list(APPEND _DEBUG_SRC_LIST ${_DEVICE_STATISTIC_LIST})
file(GLOB_RECURSE _CHECKSUM_LIST "${CMAKE_CURRENT_SOURCE_DIR}/checksum/*.cc")
list(APPEND _DEBUG_SRC_LIST ${_CHECKSUM_LIST})
if(${debug_files_size} GREATER 0)
  add_library(_mindspore_debug_obj OBJECT ${_DEBUG_SRC_LIST})
  add_dependencies(_mindspore_debug_obj proto_input)
endif()
