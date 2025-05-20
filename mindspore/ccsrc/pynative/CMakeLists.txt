file(GLOB_RECURSE _PYNATIVE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "forward/*.cc"
        "grad/function/*.cc"
        "grad/ir/*.cc"
        "grad/jit/*.cc"
        "grad/*.cc"
        "grad/ir/*.cc"
        "grad/function/*.cc"
        "op_function/*.cc"
        "op_function/auto_generate/*.cc"
        "parallel/*.cc"
        "*.cc"
)
set_property(SOURCE ${_PYNATIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_library(mindspore_pynative SHARED ${_PYNATIVE_SRC_LIST})
    target_compile_definitions(mindspore_pynative PRIVATE PYNATIVE_DLL)
else()
    add_library(_mindspore_pynative_obj OBJECT ${_PYNATIVE_SRC_LIST})
    list(APPEND PYNATIVE_OBJ_SRC $<TARGET_OBJECTS:_mindspore_pynative_obj>)
    add_library(mindspore_pynative SHARED ${PYNATIVE_OBJ_SRC})
endif()
target_link_libraries(mindspore_pynative PRIVATE mindspore_core mindspore_ops mindspore_frontend mindspore_pyboost
                      mindspore_common mindspore_ops_kernel_common mindspore_ms_backend mindspore_backend_manager
                      mindspore::pybind11_module mindspore_profiler mindspore_runtime_pipeline
                      mindspore_backend_common)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_pynative PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_pynative PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_pynative PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

add_dependencies(mindspore_pynative generated_code)
