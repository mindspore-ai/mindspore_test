if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

file(GLOB_RECURSE D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_query.cc"
    "ascend_kernel_mod.cc"
    "acl/*.cc"
    "ge/*.cc"
    "tbe/*.cc"
    "host/*.cc"
    "aicpu/*.cc"
    "rts/*.cc"
    "hccl/*.cc"
    "akg/*.cc"
    "graph_kernel/*.cc"
    "bisheng/bisheng_kernel_mod.cc"
    "bisheng/bisheng_kernel_build.cc"
    "bisheng/custom_bisheng_kernel.cc"
    "bisheng/bisheng_kernel_metadata.cc"
)

file(GLOB_RECURSE OPS_KERNEL_ASCEND_SRC_LIST "${MS_OPS_KERNEL_DIR}/ascend/*.cc")
file(GLOB_RECURSE AICPU_OPS_SRC "${MS_OPS_KERNEL_DIR}/ascend/aicpu/*.cc")
list(REMOVE_ITEM OPS_KERNEL_ASCEND_SRC_LIST ${AICPU_OPS_SRC})
list(APPEND D_SRC_LIST ${OPS_KERNEL_ASCEND_SRC_LIST})

file(GLOB_RECURSE INTERNAL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "internal/*.cc"
)

if(ENABLE_SECURITY)
    list(REMOVE_ITEM D_SRC_LIST "rts/profiling_kernel_mod.cc")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ENABLE_D)
    file(GLOB_RECURSE AKG_D_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "akg/*.cc")
endif()

file(GLOB_RECURSE SRC_IN_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "acl/*.cc"
        "ge/*.cc"
        "host/*.cc"
        "hccl/*.cc"
        "rts/*.cc"
        "akg/*.cc"
        "graph_kernel/*.cc"
        )
list(APPEND SRC_IN_910B ${OPS_KERNEL_ASCEND_SRC_LIST})
file(GLOB_RECURSE INTERNAL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "internal/*.cc"
)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    set(SRC_IN_910B ${SRC_IN_910B}
        "internal/internal_kernel_build.cc"
        "internal/internal_ascend_adapter.cc"
    )
else()
    set(SRC_IN_910B ${SRC_IN_910B}
        "internal/internal_kernel_build.cc"
        "internal/internal_ascend_adapter.cc"
    )
    list(REMOVE_ITEM SRC_IN_910B "hccl/hcom_matmul_all_reduce.cc")
endif()

if(ENABLE_DVM)
    file(GLOB_RECURSE DVM_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "dvm/*.cc"
            )
    list(APPEND SRC_IN_910B "${DVM_SRC}")
endif()

set_property(SOURCE ${D_SRC_LIST} ${AKG_D_SRC_LIST} ${SRC_IN_910B}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
if(ENABLE_D)
    add_library(_mindspore_plugin_device_ascend_kernel_910b_obj OBJECT ${SRC_IN_910B})
endif()

add_subdirectory(aicore)

add_subdirectory(${MS_OPS_KERNEL_DIR}/ascend/ascendc "ascendc")
include_directories(${MS_OPS_KERNEL_DIR}/ascend/aicpu/aicpu_ops)
add_subdirectory(${MS_OPS_KERNEL_DIR}/ascend/aicpu/aicpu_ops aicpu/aicpu_ops)
add_subdirectory(internal)
