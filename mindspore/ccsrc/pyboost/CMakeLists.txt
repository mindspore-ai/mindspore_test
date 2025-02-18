file(GLOB_RECURSE _PYBOOST_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "auto_generate/*.cc"
        "*.cc"
        "customize/*.cc"
        "functions/*.cc"
        "functions/auto_generate/*.cc"
        "grad_functions/*.cc"
        "grad_functions/auto_generate/*.cc"
)

set_property(SOURCE ${_PYBOOST_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_PYNATIVE)
add_library(mindspore_pyboost SHARED ${_PYBOOST_SRC_LIST})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_pyboost PRIVATE PYBOOST_DLL)
endif()
target_link_libraries(mindspore_pyboost PRIVATE mindspore_core mindspore_ops mindspore_common mindspore_backend
                      mindspore::pybind11_module)
add_dependencies(mindspore_pyboost generated_code)
