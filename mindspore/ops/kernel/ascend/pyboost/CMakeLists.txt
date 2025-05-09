include(${OPS_DIR}/cmake/merge_ops.cmake)

file(GLOB_RECURSE PYBOOST_SRC_LIST "*.cc")

list(REMOVE_ITEM PYBOOST_SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/atb_runner.cc
)

# merge customize pyboost kernel files
file(GLOB_RECURSE PYBOOST_MERGE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/customize/*.cc)
list(REMOVE_ITEM PYBOOST_SRC_LIST ${PYBOOST_MERGE_SRC_LIST})

set(ASCEND_PYBOOST_MERGE_DIR ${CMAKE_BINARY_DIR}/merge/mindspore/ops/kernel/ascend/pyboost)
merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR}/customize ${ASCEND_PYBOOST_MERGE_DIR} ascend_pyboost_merge "")
file(GLOB_RECURSE ASCEND_PYBOOST_MERGE_FILES ${ASCEND_PYBOOST_MERGE_DIR}/*.cc)

if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    include_directories(${ASCEND_NNAL_ATB_PATH}/include)
endif()

add_library(_mindspore_ops_kernel_ascend_pyboost_obj OBJECT ${PYBOOST_SRC_LIST} ${ASCEND_PYBOOST_MERGE_FILES})

if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    file(GLOB_RECURSE PYBOOST_ATB_SRCS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        atb_runner.cc
        ../../../../ccsrc/plugin/device/ascend/kernel/atb/atb_adapter.cc
    )

    add_library(mindspore_pyboost_atb_kernels SHARED ${PYBOOST_ATB_SRCS})
    find_library(ATB_KENREL atb ${ASCEND_NNAL_ATB_OPP_PATH})
    target_link_libraries(mindspore_pyboost_atb_kernels PRIVATE ${ATB_KENREL} mindspore_backend_common
        mindspore_pyboost mindspore_ascend)
    set_target_properties(mindspore_pyboost_atb_kernels PROPERTIES INSTALL_RPATH
        $ORIGIN:$ORIGIN/../:$ORIGIN/../../:$ORIGIN/../../../)
endif()
