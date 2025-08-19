add_subdirectory(pyboost_impl)
add_subdirectory(kernel_mod_impl)
add_library(_mindspore_ops_kernel_ascend_aclnn_obj OBJECT ${KERNEL_MOD_IMPL_SRCS} ${PYBOOST_IMPL_SRCS})