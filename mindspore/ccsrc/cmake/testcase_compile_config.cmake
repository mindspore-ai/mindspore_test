include_directories(${CMAKE_BINARY_DIR})
include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ge/ge_operator_stub.cc)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/transform/util.cc)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/pipeline/action_stub.cc)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/cluster/cluster_stub.cc)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/profiling/parallel_strategy_profiling_stub.cc)
list(APPEND STUB_COMMON_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/profiling/profiling_stub.cc)

list(APPEND EXPRESSION_STUB_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ps/ps_core_stub.cc)

add_library(stub_common STATIC ${STUB_COMMON_SOURCE})
target_link_libraries(mindspore_common PUBLIC stub_common)

add_library(expression_ STATIC ${EXPRESSION_STUB_SOURCE})
target_link_libraries(_c_expression PUBLIC expression_)

include_directories(${CMAKE_BINARY_DIR})
list(APPEND STUB_BACKEND_SOURCE ${CMAKE_SOURCE_DIR}/tests/ut/cpp/stub/ps/ps_core_stub.cc)
add_library(stub_backend_obj OBJECT ${STUB_BACKEND_SOURCE})