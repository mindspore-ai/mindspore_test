include_directories(${CMAKE_SOURCE_DIR}/mindspore/ops)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/core/mindrt/include)

set(OPS_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(OPS_DIR ${OPS_DIR} PARENT_SCOPE)

include(cmake/op_def.cmake)

# mindspore_ops
if(ONLY_BUILD_DEVICE_PLUGINS)
    add_library(mindspore_ops INTERFACE) # dummy target to link
else()
    include(cmake/ops.cmake)
endif()

# mindspore_ops_grad
file(GLOB_RECURSE GRAD_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${OPS_DIR}/grad/*.cc)
add_library(mindspore_ops_grad STATIC ${GRAD_SRC})
target_compile_definitions(mindspore_ops_grad PRIVATE COMMON_DLL BUILDING_ME_DLL)
set_property(TARGET mindspore_ops_grad PROPERTY COMPILE_DEFINITIONS
  SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)

add_subdirectory(kernel)

# mindspore_ops_host
include(cmake/ops_host.cmake)
