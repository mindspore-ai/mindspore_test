include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)

# AICPU
add_subdirectory(aicpu/aicpu_ops)

set(ASCEND_KERNEL_COMP
        pyboost
        opapi
        availability
        acl
        acl_ir
)
foreach(_comp ${ASCEND_KERNEL_COMP})
    add_subdirectory(${_comp})
    if(TARGET _mindspore_ops_kernel_ascend_${_comp}_obj)
        list(APPEND ASCEND_KERNEL_SRC $<TARGET_OBJECTS:_mindspore_ops_kernel_ascend_${_comp}_obj>)
    endif()
endforeach()

set_property(SOURCE ${ASCEND_KERNEL_SRC} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(mindspore_ops_ascend SHARED ${ASCEND_KERNEL_SRC})
if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    target_compile_definitions(mindspore_ops_ascend PRIVATE OPS_ASCEND_DLL)
endif()
target_link_libraries(mindspore_ops_ascend PRIVATE mindspore_core mindspore_ops mindspore_common
                      mindspore_ops_kernel_common mindspore_pyboost mindspore_backend mindspore_ascend_res_manager)
set_target_properties(mindspore_ops_ascend PROPERTIES INSTALL_RPATH "$ORIGIN:$ORIGIN/..")
