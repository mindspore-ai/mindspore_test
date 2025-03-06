file(GLOB_RECURSE GRAD_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OPS_DIR}/grad/*.cc)
add_library(mindspore_ops_grad STATIC ${GRAD_SRC})
target_compile_definitions(mindspore_ops_grad PRIVATE COMMON_DLL BUILDING_ME_DLL)
set_property(TARGET mindspore_ops_grad PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)
target_include_directories(mindspore_ops_grad PRIVATE ${CMAKE_SOURCE_DIR}/mindspore/ccsrc)