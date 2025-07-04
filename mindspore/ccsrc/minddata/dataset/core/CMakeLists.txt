file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
set(DATASET_CORE_SRC_FILES
        client.cc
        config_manager.cc
        cv_tensor.cc
        data_type.cc
        device_resource.cc
        device_tensor.cc
        de_tensor.cc
        global_context.cc
        tensor.cc
        tensor_helpers.cc
        tensor_row.cc
        tensor_shape.cc
        ascend_resource.cc
        )

if(ENABLE_PYTHON AND NOT WIN32 AND NOT WIN64)
    set(DATASET_CORE_SRC_FILES
        ${DATASET_CORE_SRC_FILES}
        shared_memory_queue.cc
        message_queue.cc
        )
endif()

if(ENABLE_D)
    set(DATASET_CORE_SRC_FILES
        ${DATASET_CORE_SRC_FILES}
        # ascend910b
        device_tensor_ascend910b.cc
        device_buffer.cc
        )
endif()

if(NOT MSLITE_ENABLE_ACL)
    set(DATASET_CORE_SRC_FILES
        ${DATASET_CORE_SRC_FILES}
        types.cc # in lite, src code has types.cc impl
        )
endif()

ms_protobuf_generate(EXAMPLE_SRCS EXAMPLE_HDRS example.proto)
ms_protobuf_generate(FEATURE_SRCS FEATURE_HDRS feature.proto)
add_library(core OBJECT ${DATASET_CORE_SRC_FILES} ${EXAMPLE_SRCS} ${FEATURE_SRCS})
if(ENABLE_NATIVE_PROTOBUF)
  add_dependencies(core ${PROTOBUF_LIB})
else()
  add_dependencies(core mindspore::protobuf)
endif()

if(ENABLE_PYTHON)
  target_include_directories(core PRIVATE ${pybind11_INCLUDE_DIRS})
endif()

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
  add_dependencies(core fbs_src)
endif()
