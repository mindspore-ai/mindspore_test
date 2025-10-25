file(GLOB_RECURSE _UTILS_SRC_LIST ./*.cc)

file(STRINGS "${TOP_DIR}/version.txt" VERSION)
add_definitions(-DVERSION=\"${VERSION}\")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(GLOB_RECURSE _UTILS_SIGNAL_SRC_FILES ./signal_util.cc)
    list(REMOVE_ITEM _UTILS_SRC_LIST ${_UTILS_SIGNAL_SRC_FILES})
endif()

set_property(SOURCE ${_UTILS_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_COMMON)

add_library(mindspore_common SHARED ${_UTILS_SRC_LIST})
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    target_compile_definitions(mindspore_common PRIVATE COMMON_DLL)
endif()

target_link_libraries(mindspore_common PRIVATE mindspore_core mindspore_ops proto_input
                      mindspore::securec mindspore_profiler)

set_target_properties(mindspore_common PROPERTIES INSTALL_RPATH $ORIGIN)
if(CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "Darwin")
    target_link_libraries(mindspore_common PRIVATE mindspore::pybind11_module)
endif()
