if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

file(GLOB_RECURSE MS_DEVICE_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
list(REMOVE_ITEM MS_DEVICE_910B "distribute/mpi_collective_group.cc"
        "distribute/collective_group_wrapper.cc" "distribute/mpi_pycc.cc" "ascend_host_queue.cc")

set_property(SOURCE ${MS_DEVICE_910B} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

add_library(_mindspore_plugin_device_ascend_hal_device_910b_obj OBJECT ${MS_DEVICE_910B})

if(ENABLE_MPI)
    set_property(SOURCE "distribute/mpi_pycc.cc"
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    pybind11_add_module(_ascend_mpi "distribute/mpi_pycc.cc")
    target_link_libraries(_ascend_mpi PRIVATE mindspore::pybind11_module mindspore::ompi)

    find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
    file(GLOB_RECURSE ASCEND_COLLECTIVE_LIST "distribute/mpi_collective_group.cc"
            "distribute/collective_group_wrapper.cc")
    set_property(SOURCE ${ASCEND_COLLECTIVE_LIST}
            PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(ascend_collective SHARED ${ASCEND_COLLECTIVE_LIST})
    target_link_libraries(ascend_collective PRIVATE ${HCCL} mindspore::ompi)
    target_link_libraries(_ascend_mpi PRIVATE ascend_collective)
endif()
