## common setting
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

set(MS_OPS_KERNEL_DIR "${CMAKE_SOURCE_DIR}/mindspore/ops/kernel")

if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(ENABLE_CPU)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpu_compile_config.cmake)
endif()

# gcc flag
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 \
            -Wno-delete-non-abstract-non-virtual-dtor -Wno-unused-private-field -Wno-overloaded-virtual \
            -Wno-unused-const-variable -Wno-pessimizing-move -Wno-range-loop-analysis -Wno-mismatched-tags \
            -Wno-c++11-narrowing")
endif()

# Set compile flags to ensure float compute consistency.
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-fast-math")
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto_compile_config.cmake)

add_subdirectory(frontend)
add_subdirectory(cluster)

set(PROFILER_SUB_COMP
    tools/profiler
    plugin/cpu/profiler
)

foreach(_comp ${PROFILER_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND PROFILER_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE PROFILER_DLL)
        endif()
    endif()
endforeach()

add_library(mindspore_profiler SHARED ${PROFILER_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_profiler PRIVATE mindspore_core proto_input)
set_target_properties(mindspore_profiler PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_profiler PRIVATE mindspore::pybind11_module)
endif()

add_subdirectory(utils)

set(BACKEND_COMMON_SUB_COMP
        tools
        backend/common
        )

foreach(_comp ${BACKEND_COMMON_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND BACKEND_COMMON_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE BACKEND_COMMON_DLL)
        endif()
    endif()
endforeach()


set_property(SOURCE ${BACKEND_COMMON_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_backend_common SHARED ${BACKEND_COMMON_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_backend_common PRIVATE mindspore_core mindspore_ops mindspore_common
        proto_input mindspore_memory_pool mindspore_hardware_abstract mindspore_profiler
        mindspore_runtime_pipeline mindspore_runtime_utils)


set(MS_BACKEND_SUB_COMP
        backend/ms_backend
        runtime/core
        )

foreach(_comp ${MS_BACKEND_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND MS_BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE BACKEND_DLL)
        endif()
    endif()
endforeach()


set_property(SOURCE ${MS_BACKEND_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_ms_backend SHARED ${MS_BACKEND_SUB_OBJECTS_SRC})


target_link_libraries(mindspore_ms_backend PRIVATE mindspore::pybind11_module)
target_link_libraries(mindspore_ms_backend PRIVATE mindspore_core mindspore_ops
        mindspore_common mindspore_backend_manager proto_input mindspore_hardware_abstract
        mindspore_profiler mindspore_runtime_pipeline mindspore_backend_common mindspore_memory_pool
        mindspore_runtime_utils)

target_link_libraries(mindspore_backend_common PRIVATE mindspore::securec)
target_link_libraries(mindspore_tools PRIVATE mindspore::securec)
target_link_libraries(mindspore_ms_backend PRIVATE mindspore::securec)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_backend_common PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_backend_common PROPERTIES INSTALL_RPATH @loader_path)
    set_target_properties(mindspore_ms_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ms_backend PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_backend_common PROPERTIES INSTALL_RPATH $ORIGIN)
    set_target_properties(mindspore_ms_backend PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_backend_common PRIVATE mindspore::ssl mindspore::crypto)
    target_link_libraries(mindspore_tools PRIVATE mindspore::ssl mindspore::crypto)
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(mindspore_backend_common PRIVATE mindspore::grpc++)
    else()
        target_link_libraries(mindspore_backend_common PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_backend_common PRIVATE mindspore::pybind11_module)
elseif(ENABLE_CPU AND NOT WIN32)
    target_link_libraries(mindspore_backend_common PRIVATE -Wl,--no-as-needed mindspore::pybind11_module)
endif()


# set c_expression building
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
add_subdirectory(pybind_api)
list(APPEND PYBIND_OBJ_SRC $<TARGET_OBJECTS:_mindspore_pybind_api_obj>)
pybind11_add_module(_c_expression NO_EXTRAS ${PYBIND_OBJ_SRC})

MESSAGE(STATUS "operation system is ${CMAKE_SYSTEM}")
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(_c_expression PRIVATE -Wl,-init,mindspore_log_init)
    set(ORIGIN_PATH $ORIGIN)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(_c_expression PROPERTIES MACOSX_RPATH ON)
    set(ORIGIN_PATH @loader_path)
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(ORIGIN_PATH $ORIGIN)
else()
    MESSAGE(FATAL_ERROR "other platform: ${CMAKE_SYSTEM_NAME}")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(NOT MSVC)
        target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore_ops_fallback
                -Wl,--no-whole-archive mindspore_core mindspore_frontend mindspore_hardware_abstract mindspore_ops
                mindspore_common mindspore_ms_backend mindspore_pyboost mindspore_pynative
                mindspore_backend_manager mindspore_profiler mindspore_runtime_pipeline
                mindspore_backend_common mindspore_runtime_utils mindspore_cluster
                mindspore_tools mindspore_pynative_utils)
        target_link_libraries(_c_expression PRIVATE -Wl,--no-as-needed mindspore_ops_cpu -Wl,--as-needed)
    else()
        target_link_libraries(_c_expression PRIVATE mindspore_core mindspore_ops mindspore_common
                mindspore_ms_backend mindspore_frontend mindspore_ops_fallback mindspore_pyboost
                mindspore_pynative mindspore_backend_manager mindspore_hardware_abstract mindspore_ops_cpu
                mindspore_profiler mindspore_runtime_pipeline mindspore_backend_common
                mindspore_runtime_utils mindspore_cluster mindspore_tools mindspore_pynative_utils)
        target_link_options(_c_expression PRIVATE
                "/WHOLEARCHIVE:mindspore_ops_fallback.lib"
        )
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(_c_expression PRIVATE
        -Wl,-all_load mindspore_ops_fallback proto_input
        -Wl,-noall_load mindspore_core mindspore_ops mindspore_common mindspore_ms_backend mindspore_pyboost
        mindspore_hardware_abstract mindspore_frontend mindspore_pynative mindspore_profiler mindspore_runtime_pipeline
        mindspore_backend_common mindspore_runtime_utils mindspore_backend_manager mindspore_cluster
        mindspore_tools mindspore_pynative_utils)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
else()
    target_link_libraries(_c_expression PRIVATE
            -Wl,--whole-archive mindspore_ops_fallback proto_input
            -Wl,--no-whole-archive mindspore_core mindspore_ops mindspore_common mindspore_ms_backend mindspore_frontend
            mindspore_runtime_pipeline mindspore_backend_common mindspore_pyboost mindspore_pynative
            mindspore_backend_manager mindspore_hardware_abstract mindspore_profiler mindspore_runtime_utils
            mindspore_cluster mindspore_tools mindspore_pynative_utils)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(_c_expression PRIVATE mindspore::glog)
if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(_c_expression PRIVATE mindspore_cpu)
    else()
        target_link_libraries(_c_expression PRIVATE -Wl,--no-as-needed mindspore_cpu -Wl,--as-needed)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path;@loader_path/lib;@loader_path/lib/plugin")
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
else()
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/lib:$ORIGIN/lib/plugin)
endif()

add_subdirectory(backend/backend_manager)
add_subdirectory(runtime/memory/mem_pool)
add_subdirectory(runtime/hardware_abstract)
add_subdirectory(runtime/utils)
add_subdirectory(runtime/pipeline)
add_subdirectory(pynative)
add_subdirectory(pynative/utils/runtime)
add_subdirectory(pynative/utils/pyboost)

if(ENABLE_MINDDATA)
    add_subdirectory(minddata/mindrecord)
    add_subdirectory(minddata/dataset)
endif()

if(ENABLE_GPU)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_compile_config.cmake)
endif()

if(ENABLE_D)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend_compile_config.cmake)
endif()

