# Include DART headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang/dair)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang/dart)

# Add DART directories
add_subdirectory(dart)

# Include ms_infer_backend Source files
file(GLOB _MS_INFER_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_MS_INFER_BACKEND_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)

# Create shared library
add_library(mindspore_ms_infer_backend SHARED ${_MS_INFER_BACKEND_SRC_LIST})
target_link_libraries(mindspore_ms_infer_backend PRIVATE dalang)
target_link_libraries(mindspore_ms_infer_backend PRIVATE mindspore::pybind11_module)
target_link_libraries(mindspore_ms_infer_backend PRIVATE mindspore_core mindspore_ops mindspore_ms_backend
        mindspore_common mindspore_ops_kernel_common mindspore_ascend_res_manager mindspore_backend_manager proto_input
        mindspore_res_manager mindspore_profiler mindspore_ascend mindspore_runtime_pipeline mindspore_backend_common
        mindspore_memory_pool mindspore_ops_ascend)

target_link_libraries(mindspore_ms_infer_backend PRIVATE securec)
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ms_infer_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ms_infer_backend PROPERTIES INSTALL_RPATH @loader_path:@loader_path/plugin/cpu)
else()
    set_target_properties(mindspore_ms_infer_backend PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin/cpu)
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_ms_infer_backend PRIVATE mindspore::dnnl mindspore::mkldnn nnacl
            mindspore_cpu_res_manager)
endif()

if(NOT WIN32)
    target_link_libraries(mindspore_ms_infer_backend PRIVATE mindspore::crypto)
endif()
