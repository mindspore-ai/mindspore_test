file(GLOB_RECURSE BUILDER_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc" "task/*.cc")

set_property(SOURCE ${BUILDER_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(mindspore_runtime_pipeline SHARED ${BUILDER_SRC_LIST})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
        target_compile_definitions(mindspore_runtime_pipeline PRIVATE RUNTIME_PIPELINE_DLL)
endif()
target_link_libraries(mindspore_runtime_pipeline PRIVATE mindspore_core mindspore_common mindspore_profiler
    mindspore_runtime_utils mindspore::pybind11_module)

