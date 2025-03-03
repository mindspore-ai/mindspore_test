file(GLOB_RECURSE OPAPI_SCRS *.cc)
add_library(_mindspore_ops_kernel_ascend_opapi_obj OBJECT ${OPAPI_SCRS})