file(GLOB_RECURSE PYBOOST_SCRS *.cc)
add_library(_mindspore_ops_kernel_ascend_pyboost_obj OBJECT ${PYBOOST_SCRS})