## common setting
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_BINARY_DIR})

set(MS_OPS_KERNEL_DIR "${CMAKE_SOURCE_DIR}/mindspore/ops/kernel")

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

if(ENABLE_SECURITY OR NOT ENABLE_CPU AND ENABLE_DEBUGGER)
    list(APPEND BACKEND_SUB_OBJECTS_SRC
            plugin/device/cpu/hal/profiler/cpu_profiling.cc
            plugin/device/cpu/hal/profiler/cpu_data_saver.cc)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/proto_compile_config.cmake)

## make sub objects
set(SUB_COMP
        transform/express_ir
        frontend/optimizer
        frontend/parallel
        frontend/operator
        frontend/expander
        pipeline/jit/ps
        pipeline/jit/pi
        pipeline/pynative
        pipeline/llm_boost
        pybind_api
        )

if(WIN32)
    add_compile_definitions(BUILDING_ME_DLL)
    add_compile_definitions(BUILDING_NP_DTYPE_DLL)
endif()

foreach(_comp ${SUB_COMP})
    add_subdirectory(${_comp})
    string(REPLACE "/" "_" sub ${_comp})
    if(TARGET _mindspore_${sub}_obj)
        list(APPEND SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_${sub}_obj>)
        add_dependencies(_mindspore_${sub}_obj proto_input mindspore_core mindspore_ops)
    endif()
endforeach()

set_property(SOURCE ${SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore STATIC ${SUB_OBJECTS_SRC})

set(COMMON_SUB_COMP
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
target_link_libraries(mindspore_common PRIVATE mindspore_core mindspore_ops proto_input mindspore::protobuf)
set_target_properties(mindspore_common PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_common PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore PUBLIC securec mindspore::flatbuffers)

if(NOT WIN32)
    target_link_libraries(mindspore PUBLIC dl)
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
        backend/graph_compiler
        backend/operator
        runtime/collective
        runtime/device
        runtime/graph_scheduler
        runtime/hardware
        runtime/pynative
        runtime/pipeline
        runtime/data_queue
        plugin/device/cpu/hal/device
        plugin/device/cpu/hal/hardware
        plugin/device/cpu/hal/profiler
        plugin/device/cpu/optimizer
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

set(CPU_KERNEL_OBJECT_COUNT 0)
add_subdirectory("${MS_OPS_KERNEL_DIR}/cpu" plugin/device/cpu/kernel)
foreach(number RANGE 1 ${CPU_KERNEL_OBJECT_COUNT})
    if(TARGET _mindspore_plugin_device_cpu_kernel_obj_${number})
        list(APPEND BACKEND_SUB_OBJECTS_SRC $<TARGET_OBJECTS:_mindspore_plugin_device_cpu_kernel_obj_${number}>)
        add_dependencies(_mindspore_plugin_device_cpu_kernel_obj_${number} proto_input)
        if(CMAKE_SYSTEM_NAME MATCHES "Windows")
            target_compile_definitions(_mindspore_plugin_device_cpu_kernel_obj_${number} PRIVATE BACKEND_DLL)
        endif()
    endif()
endforeach()
set_property(SOURCE ${BACKEND_SUB_OBJECTS_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_ME)
add_library(mindspore_backend SHARED ${BACKEND_SUB_OBJECTS_SRC})

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_link_libraries(mindspore_backend PRIVATE mindspore::pybind11_module)
endif()

target_link_libraries(mindspore_backend PRIVATE mindspore_core mindspore_ops
        mindspore_common proto_input mindspore::protobuf)
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
        target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::sentencepiece)
    else()
        target_link_libraries(mindspore PUBLIC -Wl,--start-group proto_input mindspore::protobuf
                mindspore::sentencepiece -Wl,--end-group)
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore PUBLIC -Wl proto_input mindspore::protobuf mindspore::sentencepiece -Wl)
else()
    target_link_libraries(mindspore PUBLIC -Wl,--start-group proto_input mindspore::protobuf -Wl,--end-group)
endif()

# set c_expression building
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set_property(SOURCE "pipeline/jit/ps/init.cc" PROPERTY
        COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PIPELINE)
pybind11_add_module(_c_expression NO_EXTRAS "pipeline/jit/ps/init.cc" NO_EXTRAS)

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
    target_link_libraries(mindspore PUBLIC mindspore::pybind11_module)
    if(NOT MSVC)
        target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore -Wl,--no-whole-archive mindspore_core
                mindspore_ops mindspore_common mindspore_backend)
    else()
        target_link_libraries(_c_expression PRIVATE mindspore_core mindspore_ops
                mindspore_common mindspore_backend mindspore)
    endif()
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::eigen mindspore::json)
    target_link_libraries(_c_expression PRIVATE -Wl,-all_load mindspore proto_input -Wl,-noall_load mindspore_core
            mindspore_ops mindspore_common mindspore_backend)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
else()
    if(ENABLE_CPU AND NOT WIN32)
        target_link_libraries(mindspore PUBLIC proto_input mindspore::protobuf mindspore::eigen mindspore::json)
    endif()
    target_link_libraries(_c_expression PRIVATE -Wl,--whole-archive mindspore proto_input -Wl,--no-whole-archive
            mindspore_core mindspore_ops mindspore_common mindspore_backend)
    target_link_libraries(_c_expression PRIVATE mindspore::pybind11_module)
endif()


set_target_properties(_c_expression PROPERTIES INSTALL_RPATH ${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin)
if(USE_GLOG)
    target_link_libraries(_c_expression PRIVATE mindspore::glog)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
    set_target_properties(_c_expression PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

if(ENABLE_CPU)
    target_link_libraries(_c_expression PRIVATE mindspore::dnnl mindspore::mkldnn nnacl)
endif()

if(ENABLE_MINDDATA)
    add_subdirectory(minddata/mindrecord)
    add_subdirectory(minddata/dataset)
endif()

if(ENABLE_TESTCASES)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/testcase_compile_config.cmake)
endif()

if(ENABLE_GPU)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gpu_compile_config.cmake)
endif()

if(ENABLE_D)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ascend_compile_config.cmake)
endif()

add_subdirectory(transform/symbol)
