file(GLOB_RECURSE _BACKEND_MANAGER_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_BACKEND_MANAGER_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_VM)
add_library(mindspore_backend_manager SHARED ${_BACKEND_MANAGER_FILES})
target_link_libraries(mindspore_backend_manager PRIVATE mindspore_core)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_backend_manager PRIVATE BACKEND_MANAGER_DLL)
endif()

set_target_properties(mindspore_backend_manager PROPERTIES INSTALL_RPATH
    ${ORIGIN_PATH}:${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin/)
