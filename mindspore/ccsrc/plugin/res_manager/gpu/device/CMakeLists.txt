include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

file(GLOB_RECURSE _GPU_DEVICE_RES_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "*.cc")
set_property(SOURCE ${_GPU_DEVICE_RES_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_res_manager_gpu_device_obj OBJECT ${_GPU_DEVICE_RES_LIST})

if(ENABLE_GPU)
    file(GLOB_RECURSE CUDA_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}  "*.cu")
    set_property(SOURCE ${CUDA_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    if(ENABLE_ASAN)
        list(REMOVE_ITEM CUDA_SRC_LIST "gpu_hash_table.cu")
    else()
        list(REMOVE_ITEM CUDA_SRC_LIST "gpu_hash_table_dummy.cu")
    endif()


    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        string(REPLACE "-arch=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        string(REPLACE "-gencode=arch=compute_53,code=sm_53;" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_70,code=sm_70)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_75,code=sm_75)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_80,code=sm_80)
        list(APPEND CUDA_NVCC_FLAGS -gencode=arch=compute_86,code=compute_86)
        list(APPEND CUDA_NVCC_FLAGS -std=c++17)
        list(REMOVE_DUPLICATES CUDA_NVCC_FLAGS)
        cuda_add_library(gpu_hash_table STATIC ${CUDA_SRC_LIST})
    endif()
endif()
