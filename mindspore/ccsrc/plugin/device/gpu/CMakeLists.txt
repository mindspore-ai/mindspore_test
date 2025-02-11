include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CUDNN_INCLUDE_PATH} ${CUDA_PATH} ${CUDA_INCLUDE_DIRS} ${CUPTI_INCLUDE_DIRS})

########### mindspore_gpu.so #####
set(GPU_SUB_COMP
        hal/device
        hal/hardware
        hal/profiler
        optimizer
        )

if(ENABLE_GPU)
    foreach(g_comp ${GPU_SUB_COMP})
        add_subdirectory(${g_comp})
        string(REPLACE "/" "_" sub ${g_comp})
        if(TARGET _mindspore_plugin_device_gpu_${sub}_obj)
            list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_gpu_${sub}_obj>)
            if(CMAKE_SYSTEM_NAME MATCHES "Windows")
                target_compile_definitions(_mindspore_plugin_device_gpu_${sub}_obj PRIVATE BUILDING_GPU_DLL)
            endif()
        endif()
    endforeach()
    list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_manager_gpu_device_obj>)
    list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_res_mananger_gpu_device_context_conf_obj>)
    list(APPEND GPU_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_gpu_res_manager_obj>)
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_plugin_res_manager_gpu_device_obj PRIVATE BUILDING_GPU_DLL)
        target_compile_definitions(_mindspore_plugin_res_mananger_gpu_device_context_conf_obj PRIVATE BUILDING_GPU_DLL)
        target_compile_definitions(_mindspore_gpu_res_manager_obj PRIVATE BUILDING_GPU_DLL)
    endif()
endif()

### cxx api need file ###
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
add_library(mindspore_gpu SHARED ${GPU_SUB_OBJECTS_SRC})
set_target_properties(mindspore_gpu PROPERTIES SOVERSION ${CUDA_VERSION})
if(MSVC)
target_link_options(mindspore_gpu PRIVATE /FORCE:UNRESOLVED)
endif()
target_link_libraries(mindspore_gpu PRIVATE mindspore_backend mindspore_core mindspore_ops mindspore_common
    mindspore_res_manager mindspore_gpu_res_manager mindspore_profiler)
target_link_libraries(mindspore_gpu PRIVATE -Wl,--no-as-needed mindspore_pyboost -Wl,--as-needed)
target_link_libraries(mindspore_gpu PRIVATE -Wl,--whole-archive mindspore_ops_gpu -Wl,--no-whole-archive)
target_link_libraries(mindspore_gpu PRIVATE securec)
set_target_properties(mindspore_gpu PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/gpu:${ORIGIN_PATH}/gpu${CUDA_VERSION}:${ORIGIN_PATH}/../:${CUDA_PATH}/lib64)
if(ENABLE_CPU)
    target_link_libraries(mindspore_gpu PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
    if(NOT WIN32)
        target_link_libraries(mindspore_gpu PRIVATE mindspore::ssl mindspore::crypto)
        target_link_libraries(mindspore_gpu PRIVATE mindspore::event mindspore::event_pthreads
            mindspore::event_openssl -Wl,--no-as-needed mindspore::event_core ps_cache)
    endif()
endif()
if(ENABLE_GPU)
    message("add gpu lib to mindspore_gpu")
    if(WIN32)
    target_link_libraries(mindspore_gpu PRIVATE
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/curand.lib
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib/x64/cudart.lib
            ${CUDA_PATH}/lib/x64/cuda.lib
            ${CUDA_PATH}/lib/x64/cusolver.lib
            ${CUDA_PATH}/lib/x64/cufft.lib
            ${CUDA_PATH}/lib/x64/cusparse.lib
            ${CUDA_PATH}/lib/x64/nvrtc.lib
            )
    else()
    target_link_libraries(mindspore_gpu PRIVATE
            ${CUBLAS_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcurand.so
            ${CUDNN_LIBRARY_PATH}
            ${CUDA_PATH}/lib64/libcudart.so
            ${CUDA_PATH}/lib64/stubs/libcuda.so
            ${CUDA_PATH}/lib64/libcusolver.so
            ${CUDA_PATH}/lib64/libcufft.so
            ${CUDA_PATH}/lib64/libcusparse.so
            ${CUDA_PATH}/lib64/libnvrtc.so
            )
    endif()

    if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND ${CUDA_VERSION} VERSION_GREATER "11.0")
        target_link_libraries(mindspore_gpu PRIVATE gpu_hash_table)
    endif()
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(ENABLE_GPU)
        target_link_libraries(mindspore_gpu PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(ENABLE_GPU)
    if(ENABLE_MPI)
        set_target_properties(nvidia_collective PROPERTIES INSTALL_RPATH ${ORIGIN_PATH})
    endif()
endif()
