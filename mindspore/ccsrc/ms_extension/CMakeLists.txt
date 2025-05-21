file(GLOB_RECURSE EXTENSION_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cc"
)
file(GLOB_RECURSE EXCLUDE_ASCEND_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/ascend "*.cc")
foreach(EXCLUDE_FILE ${EXCLUDE_ASCEND_FILES})
    list(REMOVE_ITEM EXTENSION_SRC_LIST "ascend/${EXCLUDE_FILE}")
endforeach()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set_property(SOURCE ${EXTENSION_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_EXTENSION)
add_library(mindspore_extension SHARED ${EXTENSION_SRC_LIST})

target_link_libraries(mindspore_extension mindspore_core mindspore_pynative mindspore::pybind11_module
                        mindspore_profiler mindspore_backend_common)

set_target_properties(mindspore_extension PROPERTIES INSTALL_RPATH $ORIGIN)

if(ENABLE_D)
    add_subdirectory(ascend)
    enable_directory_when_only_build_plugins(ascend)
endif()
