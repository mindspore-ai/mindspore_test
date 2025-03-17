include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include(${OPS_DIR}/cmake/merge_ops.cmake)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

file(GLOB CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "eigen/*.cc"
    "mkldnn/*.cc"
    "environ/*.cc"
    "rpc/*.cc"
    "utils/*.cc"
    "map_tensor/*.cc"
    "sequence/*.cc"
)

set(IGNORE_LIST "")

if(NOT ENABLE_MPI)
    list(APPEND IGNORE_LIST "allgather_cpu_kernel.cc")
    list(APPEND IGNORE_LIST "reduce_scatter_cpu_kernel.cc")
    list(APPEND IGNORE_LIST "embedding_look_up_comm_grad_cpu_kernel.cc")
endif()

if(NOT ENABLE_CPU OR WIN32)
    list(APPEND IGNORE_LIST "allreduce_cpu_kernel.cc")
endif()

# Convert IGNORE_LIST to regex pattern
set(EXCLUDE_PATTERN "")
foreach(IGNORE_FILE ${IGNORE_LIST})
    if(EXCLUDE_PATTERN)
        set(EXCLUDE_PATTERN "${EXCLUDE_PATTERN}|${IGNORE_FILE}")
    else()
        set(EXCLUDE_PATTERN "${IGNORE_FILE}")
    endif()
endforeach()

if(NOT BUILD_LITE)
    file(GLOB_RECURSE PYBOOST_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "pyboost/*.cc"
        "pyboost/auto_generate/*.cc"
        )
    list(APPEND CPU_SRC_LIST "${PYBOOST_CPU_SRC_LIST}")
endif()

# merge cpu kernel files, NOT recursively
if(NOT BUILD_LITE)
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows" AND NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(CPU_KERNEL_MERGE_DIR ${CMAKE_BINARY_DIR}/merge/mindspore/ops/kernel/cpu)
        merge_ops_files(${CMAKE_CURRENT_SOURCE_DIR} ${CPU_KERNEL_MERGE_DIR} cpu_kernel_merge "${EXCLUDE_PATTERN}" TRUE)
        file(GLOB_RECURSE CPU_KERNEL_MERGE_FILES ${CPU_KERNEL_MERGE_DIR}/*.cc)
        list(APPEND CPU_SRC_LIST ${CPU_KERNEL_MERGE_FILES})
    else()
        file(GLOB CPU_KERNELS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc) # no merge on windows due to out of heap space error
        foreach(IGNORE_FILE ${IGNORE_LIST})
            foreach(KERNEL_FILE ${CPU_KERNELS})
                if(${KERNEL_FILE} MATCHES ${IGNORE_FILE})
                    list(REMOVE_ITEM CPU_KERNELS ${KERNEL_FILE})
                endif()
            endforeach()
        endforeach()
        list(APPEND CPU_SRC_LIST ${CPU_KERNELS})
    endif()
endif()

if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM)
endif()

if(BUILD_LITE)
    # mslite do not support python op
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maxv ")
    string(REPLACE "-Wall" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-Wall" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

set_property(SOURCE ${CPU_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)

if(ENABLE_CPU)
    add_library(_mindspore_ops_cpu_kernel_obj OBJECT ${CPU_SRC_LIST})
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_ops_cpu_kernel_obj PRIVATE OPS_HOST_DLL)
    endif()
endif()
