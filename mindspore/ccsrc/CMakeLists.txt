## common setting
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/device/res_manager)
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

# for pybind_api/np_dtype/np_dtypes.cc
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    find_package(Python3 COMPONENTS NumPy Development)
    if(Python3_NumPy_FOUND)
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
    endif()
endif()

## make sub objects
set(SUB_COMP
        transform/express_ir
        frontend/optimizer
        frontend/parallel
        frontend/operator
        frontend/expander
        frontend/ir
        pipeline/jit/ps
        pipeline/jit/pi
        pipeline/jit/trace
        pipeline/llm_boost
        )

foreach(_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core mindspore_ops mindspore_backend)
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(_mindspore_${sub}_obj PRIVATE FRONTEND_DLL BUILDING_ME_DLL BUILDING_NP_DTYPE_DLL)
    endif()
endforeach()

set(PYBIND_SRC
    pybind_api/np_dtype/np_dtypes.cc
)
list(APPEND SUB_OBJECTS_SRC ${PYBIND_SRC})

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_frontend SHARED ${SUB_OBJECTS_SRC})
target_link_libraries(mindspore_frontend PRIVATE mindspore_backend mindspore_core mindspore::pybind11_module
                      mindspore_pyboost mindspore_common mindspore_ops mindspore_backend_manager proto_input)
# for PYBIND_SRC
target_compile_definitions(mindspore_frontend PRIVATE FRONTEND_DLL BUILDING_ME_DLL BUILDING_NP_DTYPE_DLL)

set(COMMON_SUB_COMP
        availability
        utils
        common
        common/debug
        common/expander
        common/symbol_engine
        common/amp
        )

foreach(_comp ${COMMON_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND COMMON_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE COMMON_DLL)
        endif()
    endif()
endforeach()


add_library(mindspore_common SHARED ${COMMON_SUB_OBJECTS_SRC})
target_link_libraries(mindspore_common PRIVATE mindspore_core mindspore_ops proto_input securec)
set_target_properties(mindspore_common PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_common PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore_frontend PUBLIC securec mindspore::flatbuffers)

if(NOT WIN32)
    target_link_libraries(mindspore_frontend PUBLIC dl)
endif()

set(BACKEND_SUB_COMP
        debug
        debug/profiler
        ps
        distributed
        kernel
        backend/common/mem_reuse
        backend/common/optimizer
        backend/common/pass
        backend/common/session
        backend/common/somas
        backend/common/graph_kernel
        backend/common/expander
        backend/ms_backend
        backend/graph_compiler
        backend/operator
        memory/mem_pool
        runtime/collective
        runtime/device
        runtime/graph_scheduler
        runtime/hardware
        runtime/pynative
        runtime/pipeline
        runtime/data_queue
        runtime/runtime_conf
        plugin/device/cpu/hal/device
        plugin/device/cpu/hal/hardware
        plugin/device/cpu/hal/profiler
        plugin/device/cpu/optimizer
        plugin/device/cpu/kernel
        plugin/res_manager/cpu
        plugin/res_manager/cpu/cpu_device_address
        plugin/res_manager/cpu/cpu_mem_manager
        )

foreach(_comp ${BACKEND_SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_${sub}_obj PRIVATE BACKEND_DLL)
        endif()
    endif()
endforeach()


set_property(SOURCE ${BACKEND_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_backend SHARED ${BACKEND_SUB_OBJECTS_SRC})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore_backend PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore_backend PRIVATE mindspore_core mindspore_ops
        mindspore_common mindspore_backend_manager proto_input mindspore_res_manager)
target_link_libraries(mindspore_backend PRIVATE securec)


if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_backend PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_backend PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_backend PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_backend PRIVATE mindspore::ssl mindspore::crypto)
endif()

if(ENABLE_DEBUGGER)
    # debugger: link grpc
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(mindspore_backend PRIVATE mindspore::grpc++)
    else()
        target_link_libraries(mindspore_backend PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_backend PRIVATE mindspore::event mindspore::event_pthreads mindspore::event_openssl
            mindspore::event_core ps_cache)
elseif(ENABLE_CPU AND NOT WIN32)
    target_link_libraries(mindspore_backend PRIVATE mindspore::event mindspore::event_pthreads mindspore::event_openssl
            -Wl,--no-as-needed mindspore::event_core ps_cache)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(MSVC)
        target_link_libraries(mindspore_frontend PUBLIC proto_input mindspore::protobuf)
    else()
        target_link_libraries(mindspore_frontend PUBLIC -Wl,--start-group proto_input mindspore::protobuf
                -Wl,--end-group)
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_frontend PUBLIC -Wl proto_input mindspore::protobuf -Wl)
else()
    target_link_libraries(mindspore_frontend PUBLIC -Wl,--start-group proto_input mindspore::protobuf -Wl,--end-group)
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
    target_link_libraries(mindspore_frontend PUBLIC mindspore::pybind11_module)
    if(NOT MSVC)
        target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore_ops_grad -Wl,--no-whole-archive
                mindspore_core mindspore_frontend mindspore_res_manager
                mindspore_ops mindspore_common mindspore_backend mindspore_pyboost mindspore_pynative
                mindspore_backend_manager)
        target_link_libraries(_c_expression PRIVATE -Wl,--no-as-needed mindspore_ops_host -Wl,--as-needed)
    else()
        target_link_libraries(_c_expression PRIVATE mindspore_core mindspore_ops
                mindspore_common mindspore_backend mindspore_frontend mindspore_ops_grad mindspore_pyboost
                mindspore_pynative mindspore_backend_manager mindspore_res_manager mindspore_ops_host)
        target_link_options(_c_expression PRIVATE "/WHOLEARCHIVE:mindspore_ops_grad.lib")
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_frontend PUBLIC proto_input mindspore::protobuf mindspore::eigen mindspore::json)
    target_link_libraries(_c_expression PRIVATE -Wl,-all_load mindspore_ops_grad proto_input -Wl,-noall_load
        mindspore_core mindspore_ops mindspore_common mindspore_backend mindspore_pyboost mindspore_backend_manager
        mindspore_res_manager mindspore_frontend mindspore_pynative)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
else()
    if(ENABLE_CPU AND NOT WIN32)
        target_link_libraries(mindspore_frontend PUBLIC proto_input mindspore::protobuf mindspore::eigen
                mindspore::json)
    endif()
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore_ops_grad proto_input
            -Wl,--no-whole-archive mindspore_core mindspore_ops mindspore_common mindspore_backend mindspore_frontend
            mindspore_pyboost mindspore_pynative mindspore_backend_manager mindspore_res_manager)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
endif()

if(ENABLE_D OR ENABLE_ACL)
    target_link_libraries(_c_expression PRIVATE -Wl,--no-whole-archive mindspore_ge_backend
        mindspore_ascend_res_manager)
    target_link_libraries(mindspore_backend PRIVATE mindspore_ascend_res_manager)
endif()


target_link_libraries(_c_expression PRIVATE mindspore::glog)
set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin)

#if(USE_GLOG) default is on
#    target_link_libraries(_c_expression PRIVATE mindspore::glog)
#endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

if(ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

add_subdirectory(pyboost)
add_subdirectory(backend/backend_manager)
add_subdirectory(runtime/device/res_manager)
add_subdirectory(pynative)

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
