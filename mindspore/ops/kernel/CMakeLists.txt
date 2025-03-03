include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops/kernel/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/include)
include_directories(${CMAKE_BINARY_DIR})

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/debug/debugger")
endif()

if(NOT BUILD_LITE)
    add_subdirectory(common)
endif()

# ===== CPU
add_subdirectory(cpu)
set(CPU_KERNEL_OBJECT_COUNT ${CPU_KERNEL_OBJECT_COUNT} PARENT_SCOPE)

# ===== Ascend
if(ENABLE_D)
    add_subdirectory(ascend)
endif()

# ===== GPU
if(ENABLE_GPU)
    add_subdirectory(gpu)
    enable_directory_when_only_build_plugins(gpu)
endif()