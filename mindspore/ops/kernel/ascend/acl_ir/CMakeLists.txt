file(GLOB_RECURSE _ACL_IR_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_ACL_IR_SRC_LIST} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_GE_ADPT)
add_library(_mindspore_ops_kernel_ascend_acl_ir_obj OBJECT ${_ACL_IR_SRC_LIST})