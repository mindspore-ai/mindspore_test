if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwins")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

if(ENABLE_CPU)
    file(GLOB_RECURSE _CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM _CPU_SRC_LIST
        "kernel_executor/mpi/mpi_adapter.cc"
        "kernel_executor/mpi/mpi_export.cc"
        "res_manager/collective/mpi_collective_comm_lib.cc"
        "res_manager/collective/mpi_communication_group.cc"
        "profiler/cpu_profiling.cc"
        "profiler/cpu_data_saver.cc")
    if(WIN32 OR APPLE)
        list(REMOVE_ITEM _CPU_SRC_LIST
            "res_manager/collective/ms_collective_comm_lib.cc"
            "res_manager/collective/allreduce_impl.cc"
            "res_manager/collective/ms_collective_ops_impl.cc"
            "res_manager/collective/ms_collective_topo.cc"
            "res_manager/collective/ms_collective_node.cc")
    endif()

    if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    else()
        file(GLOB_RECURSE AKG_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "kernel_executor/akg/*.cc"
            "kernel_executor/dynamic_akg/*.cc"
        )
        list(REMOVE_ITEM _CPU_SRC_LIST ${AKG_CPU_SRC_LIST})
    endif()

    if(ENABLE_MPI)
        set(_CPU_MPI_SRC_LIST "kernel_executor/mpi/mpi_adapter.cc" "kernel_executor/mpi/mpi_export.cc")
        set_property(SOURCE ${_CPU_MPI_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(mpi_adapter SHARED ${_CPU_MPI_SRC_LIST})
        target_link_libraries(mpi_adapter PRIVATE mindspore::ompi mindspore::pybind11_module -ldl mindspore::securec)

        set(_CPU_MPI_COLLECTIVE_SRC_LIST
            "res_manager/collective/mpi_collective_comm_lib.cc"
            "res_manager/collective/mpi_communication_group.cc"
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
        set_property(SOURCE ${_CPU_MPI_COLLECTIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
        add_library(mpi_collective SHARED ${_CPU_MPI_COLLECTIVE_SRC_LIST})
        target_link_libraries(mpi_collective PRIVATE mindspore::ompi)
    endif()

    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
    endif()

    set_property(SOURCE ${_CPU_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(_mindspore_plugin_cpu_obj OBJECT ${_CPU_SRC_LIST})
endif()
