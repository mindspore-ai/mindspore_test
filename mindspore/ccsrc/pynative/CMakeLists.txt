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
        "*.cc"
)
set_property(SOURCE ${_PYNATIVE_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(mindspore_pynative SHARED ${_PYNATIVE_SRC_LIST})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_pynative PRIVATE PYNATIVE_DLL)
endif()
target_link_libraries(mindspore_pynative mindspore_core mindspore_ops mindspore_frontend mindspore_pyboost
                      mindspore_common mindspore_backend mindspore::pybind11_module)
add_dependencies(mindspore_pynative generated_code)
