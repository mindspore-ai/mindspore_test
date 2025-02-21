include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)

# AICPU
add_subdirectory(aicpu/aicpu_ops)

# PyBoost & Aclnn
file(GLOB_RECURSE PYBOOST_SCRS pyboost/*.cc)
file(GLOB_RECURSE OPAPI_SCRS opapi/*.cc)
file(GLOB_RECURSE AVAIL_SCRS availability/*.cc)
set(ASCEND_KERNEL_SRC ${PYBOOST_SCRS} ${OPAPI_SCRS} ${AVAIL_SCRS})
set_property(SOURCE ${ASCEND_KERNEL_SRC}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(mindspore_ops_ascend STATIC ${ASCEND_KERNEL_SRC})