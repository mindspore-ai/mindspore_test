include(${CMAKE_SOURCE_DIR}/cmake/gpu_env_setup.cmake)

if(GPU_BACKEND_CUDA)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/gpu)
    add_subdirectory(plugin/device/gpu)
    enable_directory_when_only_build_plugins(plugin/device/gpu)
endif()

if(GPU_BACKEND_ROCM)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/plugin/device/amd)
    add_subdirectory(plugin/device/amd)
    enable_directory_when_only_build_plugins(plugin/device/amd)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(_c_expression PRIVATE mindspore::ssl mindspore::crypto)
endif()
