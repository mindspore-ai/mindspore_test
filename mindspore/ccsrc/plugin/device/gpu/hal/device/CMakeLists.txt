if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

list(APPEND DEVICE_SRC_LIST "ps/gpu_ps_cache.cc")
if(ENABLE_GPU)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/mem_reuse/mem_reuse.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/backend/common/mem_reuse/mem_swap_manager.cc)
    list(APPEND DEVICE_SRC_LIST ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/data_queue/data_queue.h)
    file(GLOB_RECURSE DEVICE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    list(REMOVE_ITEM DEVICE_SRC_LIST
            "distribution/collective_wrapper.cc"
            "distribution/mpi_wrapper.cc"
            "distribution/nccl_wrapper.cc"
            "trt_loader.cc")
    if(NOT ${TENSORRT_HOME} STREQUAL "")
        find_path(TENSORRT_HOME_INCLUDE NvInfer.h HINTS ${TENSORRT_HOME}/include)
        if(TENSORRT_HOME_INCLUDE STREQUAL TENSORRT_HOME_INCLUDE-NOTFOUND)
            message(FATAL_ERROR "Tensor-RT dir not exist ${TENSORRT_HOME}")
        endif()
        message("Enable GPU inference. Tensor-RT include dir: ${TENSORRT_HOME_INCLUDE}")
        set(ENABLE_GPU_INFER TRUE)
        add_compile_definitions(ENABLE_GPU_INFER)
        include_directories(${TENSORRT_HOME_INCLUDE})
        list(APPEND DEVICE_SRC_LIST ${CMAKE_CURRENT_SOURCE_DIR}/trt_loader.cc)
    endif()
endif()

set_property(SOURCE ${DEVICE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(_mindspore_plugin_device_gpu_hal_device_obj OBJECT ${DEVICE_SRC_LIST})
