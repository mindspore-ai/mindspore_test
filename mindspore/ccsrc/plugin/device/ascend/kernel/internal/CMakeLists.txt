file(GLOB_RECURSE INTERNAL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)
list(REMOVE_ITEM INTERNAL_SRC_LIST
    "internal_kernel_build.cc"
    "internal_ascend_adapter.cc"
)

set(ME_KERNELS_INTERNAL_LIB_PATH
    ${ORIGIN_PATH}/ms_kernels_internal/internal_kernel/lib)

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME})
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/include)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/lccl/include)

    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/lib)
    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/lccl/lib)

    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
    add_library(mindspore_internal_kernels SHARED ${INTERNAL_SRC_LIST})

    target_link_libraries(mindspore_internal_kernels PRIVATE mindspore_ascend ms_kernels_internal)
    set_target_properties(mindspore_internal_kernels PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/ascend/:${ORIGIN_PATH}/../:${ME_KERNELS_INTERNAL_LIB_PATH}:${ASCEND_RPATH})
    set_target_properties(lowlatency_collective PROPERTIES INSTALL_RPATH
                    ${ORIGIN_PATH}/ms_kernels_internal/lccl/lib)
endif()
