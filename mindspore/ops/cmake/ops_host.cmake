include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/include)
include_directories(${CMAKE_BINARY_DIR})

# CPU kernel objects
if(ENABLE_CPU)
    if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "aarch64")
        set(PLATFORM_ARM64 "on")
        set(X86_64_SIMD "off")
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set(X86_64_SIMD "off")
    elseif("${X86_64_SIMD}" STREQUAL "off" AND NOT ${ENABLE_ASAN})
        set(X86_64_SIMD "avx512")
    endif()
    add_subdirectory(kernel/cpu/nnacl)
endif()

set(CPU_KERNEL_OBJECTS "")
foreach(number RANGE 1 ${CPU_KERNEL_OBJECT_COUNT})
    if(TARGET _mindspore_ops_cpu_kernel_obj_${number})
        list(APPEND CPU_KERNEL_OBJECTS $<TARGET_OBJECTS:_mindspore_ops_cpu_kernel_obj_${number}>)
        add_dependencies(_mindspore_ops_cpu_kernel_obj_${number} proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_ops_cpu_kernel_obj_${number} PRIVATE OPS_HOST_DLL)
        endif()
    endif()
endforeach()

if((NOT ENABLE_CPU) OR (BUILD_LITE OR ENABLE_TESTCASES))
    add_library(mindspore_ops_host INTERFACE)
else()
    add_library(mindspore_ops_host SHARED ${CPU_KERNEL_OBJECTS})
    target_link_libraries(mindspore_ops_host PRIVATE mindspore_core mindspore_ops mindspore_memory_pool
        mindspore_common mindspore_ops_kernel_common mindspore_ms_backend mindspore_pyboost mindspore_profiler
        mindspore_runtime_pipeline mindspore_backend_common nnacl mindspore::dnnl mindspore::mkldnn)
    target_link_libraries(mindspore_ops_host PRIVATE securec)

    set_target_properties(mindspore_ops_host PROPERTIES INSTALL_RPATH
            "$ORIGIN:$ORIGIN/..")
endif()