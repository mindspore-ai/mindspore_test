## make protobuf files
file(GLOB ONNX_PROTO "" ${CMAKE_SOURCE_DIR}/third_party/proto/onnx/onnx.proto)
message("onnx proto path is :" ${ONNX_PROTO})
ms_protobuf_generate(ONNX_PROTO_SRCS ONNX_PROTO_HDRS ${ONNX_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${ONNX_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/ps/core")
file(GLOB_RECURSE COMM_PROTO_IN RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "ps/core/protos/*.proto")
ms_protobuf_generate(COMM_PROTO_SRCS COMM_PROTO_HDRS ${COMM_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${COMM_PROTO_SRCS})

include_directories("${CMAKE_BINARY_DIR}/distributed/cluster/topology")
file(GLOB_RECURSE DISTRIBUTED_CLUSTER_TOPOLOGY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "distributed/cluster/topology/protocol/*.proto")
ms_protobuf_generate(DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS DISTRIBUTED_CLUSTER_TOPOLOGY_HDRS
        ${DISTRIBUTED_CLUSTER_TOPOLOGY})
list(APPEND MINDSPORE_PROTO_LIST ${DISTRIBUTED_CLUSTER_TOPOLOGY_SRCS})

if(NOT ENABLE_SECURITY)
    file(GLOB_RECURSE PROFILER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "plugin/device/ascend/hal/profiler/memory_profiling.proto")
    ms_protobuf_generate_py(PROFILER_MEM_PROTO_PY PROFILER_MEM_PROTO_HDRS_PY PROFILER_MEM_PROTO_PYS
            ${PROFILER_PROTO_LIST})
    list(APPEND MINDSPORE_PROTO_LIST ${PROFILER_MEM_PROTO_PY})
endif()

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/debug/debugger")
    file(GLOB_RECURSE DEBUGGER_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/debug_graph.proto")
    ms_protobuf_generate(DEBUGGER_PROTO_SRCS DEBUGGER_PROTO_HDRS ${DEBUGGER_PROTO_LIST})
    file(GLOB_RECURSE DUMP_DATA_PROTO_LIST FOLLOW_SYMLINKS "${ASCEND_PATH}/latest/include/proto/dump_data.proto")
    list(LENGTH DUMP_DATA_PROTO_LIST PROTO_LEN)
    if(PROTO_LEN EQUAL 0)
        message("Can't find dump_data.proto in path: ${ASCEND_PATH}/latest/include/proto, use proto in mindspore path.")
        file(GLOB_RECURSE DUMP_DATA_PROTO_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "debug/debugger/dump_data.proto")
    endif()
    ms_protobuf_generate(DUMP_DATA_PROTO_SRCS DUMP_DATA_PROTO_HDRS ${DUMP_DATA_PROTO_LIST})
    list(APPEND MINDSPORE_PROTO_LIST ${DEBUGGER_PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${DUMP_DATA_PROTO_SRCS})
endif()

if(ENABLE_DUMP_PROTO)
    include_directories(${CMAKE_BINARY_DIR})

    file(GLOB_RECURSE PROTO_PY RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "utils/anf_ir.proto"
            "utils/summary.proto"
            "utils/lineage.proto"
            "utils/checkpoint.proto"
            "utils/print.proto"
            "utils/node_strategy.proto"
            "utils/profiling_parallel.proto"
            )
    ms_protobuf_generate_py(PY_SRCS PY_HDRS PY_PYS ${PROTO_PY})

    list(APPEND MINDSPORE_PROTO_LIST ${PROTO_SRCS})
    list(APPEND MINDSPORE_PROTO_LIST ${PY_SRCS})
endif()

# core/mindir.proto
file(GLOB_RECURSE CORE_PROTO_IN ${CMAKE_SOURCE_DIR}/mindspore/core/proto/*.proto)
ms_protobuf_generate(CORE_PROTO_SRC CORE_PROTO_HDR ${CORE_PROTO_IN})
list(APPEND MINDSPORE_PROTO_LIST ${CORE_PROTO_SRC})

include_directories("${CMAKE_BINARY_DIR}/runtime/graph_scheduler/actor/rpc")
file(GLOB_RECURSE RPC_PROTO RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "runtime/graph_scheduler/actor/rpc/*.proto")
ms_protobuf_generate(RPC_PROTO_SRCS RPC_PROTO_HDRS ${RPC_PROTO})
list(APPEND MINDSPORE_PROTO_LIST ${RPC_PROTO_SRCS})

if(MINDSPORE_PROTO_LIST)
    add_library(proto_input STATIC ${MINDSPORE_PROTO_LIST})
    if(NOT MSVC)
        set_target_properties(proto_input PROPERTIES COMPILE_FLAGS "-Wno-unused-variable -Wno-array-bounds")
    endif()
endif()