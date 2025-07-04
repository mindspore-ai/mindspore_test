include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)

### check dvm binary file ###
set(DVM_LIB)
if(ENABLE_DVM)
    set(DVM_LIB $ENV{DVM_LIB})
    message(STATUS "DVM module is enabled")
    message(STATUS "DVM library file: ${DVM_LIB}")
else()
    message(WARNING "The binary files tracked by git lfs have not been downloaded, and Graph Kernel Fusion can not "
            "be enabled on Ascend! Please perform the following steps:\n"
            "1. Install git lfs, refer https://github.com/git-lfs/git-lfs/wiki/installation\n"
            "2. After installing git lfs, do not forget executing the following command:\n"
            "   git lfs install\n"
            "3. Download the files tracked by git lfs, executing the following commands:\n"
            "   cd ${CMAKE_SOURCE_DIR}\n"
            "   git lfs pull\n"
            "4. Re-compile the source codes")
endif()

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
endif()


########### mindspore_ascend.so #####
set(ASCEND_SUB_COMP
        hal/device
        hal/hardware
        kernel
        optimizer
        llm_boost
        )

foreach(a_comp ${ASCEND_SUB_COMP})
    add_subdirectory(${a_comp})
    string(REPLACE "/" "_" sub ${a_comp})
    if(TARGET _mindspore_plugin_device_ascend_${sub}_obj)
        list(APPEND ASCEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_ascend_${sub}_obj>)
        add_dependencies(_mindspore_plugin_device_ascend_${sub}_obj proto_input)
    endif()
    if(TARGET _mindspore_plugin_device_ascend_${sub}_910b_obj)
        list(APPEND ASCEND_910B_OBJ_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_ascend_${sub}_910b_obj>)
        add_dependencies(_mindspore_plugin_device_ascend_${sub}_910b_obj proto_input)
    endif()
endforeach()

set(ASCEND_COMMON_SUB
        hal/profiler
        hal/special
        )

foreach(a_comp ${ASCEND_COMMON_SUB})
    add_subdirectory(${a_comp})
    string(REPLACE "/" "_" sub ${a_comp})
    if(TARGET _mindspore_plugin_device_ascend_${sub}_obj)
        list(APPEND ASCEND_COMMON_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_ascend_${sub}_obj>)
        add_dependencies(_mindspore_plugin_device_ascend_${sub}_obj proto_input)
    endif()
endforeach()

list(APPEND ASCEND_910B_OBJ_SRC ${ASCEND_COMMON_OBJECTS_SRC})
add_library(mindspore_ascend SHARED ${ASCEND_910B_OBJ_SRC})
set_target_properties(mindspore_ascend PROPERTIES VERSION 2)

set(MS_KERNELS_INTERNAL_LIB_PATH
    ${ORIGIN_PATH}/ascend/ms_kernels_internal/internal_kernel/lib)

target_link_libraries(mindspore_ascend PRIVATE mindspore_backend_common mindspore_core mindspore_ops mindspore_common
                      mindspore_ops_kernel_common mindspore_ops_ascend mindspore_profiler mindspore_runtime_pipeline
                      mindspore_ms_backend)
target_link_libraries(mindspore_ascend PRIVATE mindspore_ascend_res_manager)
target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed mindspore_pyboost -Wl,--as-needed)
target_link_libraries(mindspore_ascend PRIVATE proto_input mindspore::protobuf)
target_link_libraries(mindspore_ascend PRIVATE securec d_collective)
target_link_libraries(mindspore_ascend PRIVATE multi_ascend_collective)
target_link_libraries(mindspore_ascend PRIVATE mindspore::ssl mindspore::crypto)
set_target_properties(mindspore_ascend PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/ascend:${ORIGIN_PATH}/../:${MS_KERNELS_INTERNAL_LIB_PATH})

include(${CMAKE_SOURCE_DIR}/cmake/dependency_graphengine.cmake)

if(DVM_LIB)
    target_link_libraries(mindspore_ascend PRIVATE ${DVM_LIB})
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    target_link_libraries(mindspore_ascend PRIVATE -Wl,--no-as-needed mindspore::grpc++)
endif()

find_library(GE_RUNNER ge_runner ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(GRAPH graph ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
target_link_libraries(mindspore_ascend PRIVATE ${GE_RUNNER} ${GRAPH})
if(ENABLE_CPU)
    target_link_libraries(mindspore_ascend PRIVATE mindspore::event mindspore::event_pthreads
        mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()

MESSAGE("USE DAV LIB PATH: ${ASCEND_PATH}")
find_library(ERROR_MANAGER error_manager ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(RUNTIME_LIB runtime ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(TSDCLIENT tsdclient HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(DATATRANSFER datatransfer HINTS ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ACL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ACL_DVPP acl_dvpp ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ACL_NNOPBASE nnopbase ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ACL_DVPP_OP acl_dvpp_op ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(ACL_DVPP_MPI acl_dvpp_mpi ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(PLATFORM platform ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(PROFILING msprofiler ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(OPT_FEATURE opt_feature ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(adump_server libadump_server.a ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(CANN_KB cann_kb ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(COMPRESS compress ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(OPSKERNEL opskernel ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(AICPU_ASCEND_ENGINE aicpu_ascend_engine ${ASCEND_CANN_PLUGIN_PATH} ${ASCEND_TOOLKIT_PLUGIN_PATH})
find_library(HOST_CPU_ENGINE host_cpu_engine ${ASCEND_CANN_PLUGIN_PATH} ${ASCEND_TOOLKIT_PLUGIN_PATH})
find_library(AICPU_TF_ENGINE aicpu_tf_engine ${ASCEND_CANN_PLUGIN_PATH} ${ASCEND_TOOLKIT_PLUGIN_PATH})
find_library(AOE_TUNING libaoe_tuning.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(SLICE slice ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
find_library(AOE_EXECUTOR aoe_executor ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})

target_link_libraries(mindspore_ascend PRIVATE ${RUNTIME_LIB} ${TSDCLIENT} ${DATATRANSFER} ${ERROR_MANAGER}
        -Wl,--no-as-needed ${PLATFORM} ${ACL} ${OPT_FEATURE}
        ${PROFILING} ${CANN_KB} ${COMPRESS} ${OPSKERNEL} ${HOST_CPU_ENGINE} ${SLICE} ${AOE_EXECUTOR}
        ${ACL_DVPP_OP} ${ACL_DVPP_MPI})
target_link_libraries(mindspore_ascend PRIVATE ${adump_server})

if(ENABLE_MPI)
    set_target_properties(_ascend_mpi PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin/)
endif()
