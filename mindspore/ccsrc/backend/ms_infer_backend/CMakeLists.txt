# Include DART headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang/dair)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/dart/dalang/dart)

# Add DART directories
add_subdirectory(dart/dalang)

# Include ms_infer_backend Source files
file(GLOB _MS_INFER_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_MS_INFER_BACKEND_SRC_LIST}
        PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)

# Create shared library
add_library(mindspore_ms_infer_backend SHARED ${_MS_INFER_BACKEND_SRC_LIST})
target_link_libraries(mindspore_ms_infer_backend PUBLIC dalang)
target_link_libraries(mindspore_ms_infer_backend PRIVATE mindspore_core mindspore_ops mindspore_common
                                                         mindspore_backend_common mindspore_backend_manager)

# Set the output directory for the shared library
if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ms_infer_backend PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ms_infer_backend PROPERTIES INSTALL_RPATH @loader_path:@loader_path/plugin)
else()
    set_target_properties(mindspore_ms_infer_backend PROPERTIES INSTALL_RPATH $ORIGIN:$ORIGIN/plugin)
endif()
