file(GLOB_RECURSE _AVAIL_SRC_LIST ./*.cc)
set_property(SOURCE ${_AVAIL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_UTILS)
add_library(_mindspore_availability_obj OBJECT ${_AVAIL_SRC_LIST})
