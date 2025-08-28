file(GLOB_RECURSE FALLBACK_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OPS_DIR}/fallback/*.cc)
add_library(mindspore_ops_fallback STATIC ${FALLBACK_SRC})
target_compile_definitions(mindspore_ops_fallback PRIVATE BACKEND_COMMON_DLL)
set_property(TARGET mindspore_ops_fallback PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)
target_include_directories(mindspore_ops_fallback PRIVATE ${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
