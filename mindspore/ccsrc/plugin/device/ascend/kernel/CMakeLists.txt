if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

file(GLOB_RECURSE SRC_IN_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "acl/*.cc"
        "ge/*.cc"
        "host/*.cc"
        "hccl/*.cc"
        "rts/*.cc"
        "akg/*.cc"
        "graph_kernel/*.cc"
        "internal/internal_kernel_build.cc"
        "internal/internal_ascend_adapter.cc"
        )

if(NOT DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    # only support in internal kernel
    list(REMOVE_ITEM SRC_IN_910B "hccl/hcom_matmul_all_reduce.cc")
endif()

file(GLOB_RECURSE OPS_KERNEL_ASCEND_SRC_LIST "${MS_OPS_KERNEL_DIR}/ascend/*.cc")
file(GLOB_RECURSE AICPU_OPS_SRC "${MS_OPS_KERNEL_DIR}/ascend/aicpu/*.cc")
list(REMOVE_ITEM OPS_KERNEL_ASCEND_SRC_LIST ${AICPU_OPS_SRC})
list(APPEND SRC_IN_910B ${OPS_KERNEL_ASCEND_SRC_LIST})

if(ENABLE_DVM)
    file(GLOB_RECURSE DVM_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "dvm/*.cc")
    list(APPEND SRC_IN_910B "${DVM_SRC}")
endif()

set_property(SOURCE ${SRC_IN_910B}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

add_library(_mindspore_plugin_device_ascend_kernel_910b_obj OBJECT ${SRC_IN_910B})

add_subdirectory(aicore)

add_subdirectory(${MS_OPS_KERNEL_DIR}/ascend/ascendc "ascendc")
include_directories(${MS_OPS_KERNEL_DIR}/ascend/aicpu/aicpu_ops)
add_subdirectory(${MS_OPS_KERNEL_DIR}/ascend/aicpu/aicpu_ops aicpu/aicpu_ops)
add_subdirectory(internal)
