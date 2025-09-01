file(GLOB_RECURSE HCCL_KENREL_SRC ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    # only support in internal kernel
    list(REMOVE_ITEM HCCL_KENREL_SRC "hcom_matmul_all_reduce.cc")
endif()

set_property(SOURCE ${HCCL_KENREL_SRC}
    PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

add_library(_mindspore_ops_kernel_ascend_hccl_obj OBJECT ${HCCL_KENREL_SRC})

