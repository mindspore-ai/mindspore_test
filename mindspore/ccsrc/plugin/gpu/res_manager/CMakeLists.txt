include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

file(GLOB_RECURSE _GPU_RES_MANAGER_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
list(REMOVE_ITEM _GPU_RES_MANAGER_LIST
    "collective/nvidia_collective_comm_lib.cc"
    "collective/nvidia_communication_group.cc")

if(ENABLE_MPI)
    set(NVIDIA_COLLECTIVE_SRC
        "collective/nvidia_collective_comm_lib.cc"
        "collective/nvidia_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/hardware_abstract/collective/communication_group.cc)
    set_property(SOURCE ${NVIDIA_COLLECTIVE_SRC} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(nvidia_collective SHARED ${NVIDIA_COLLECTIVE_SRC})
    target_link_libraries(nvidia_collective PRIVATE mindspore::nccl)
endif()

if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  "*.cu")
    set_property(SOURCE ${CUDA_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)

    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        string(REPLACE "-arch=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        string(REPLACE "-gencode=arch=compute_53,code=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_70,code=sm_70)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code=compute_86)
        list(APPEND CUDA_NVCC_FLAGS -std=c++17)
        list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
    endif()
endif()

set_property(SOURCE ${_GPU_RES_MANAGER_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_gpu_res_manager_obj OBJECT ${_GPU_RES_MANAGER_LIST})
