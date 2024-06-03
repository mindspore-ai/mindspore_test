include_directories(${OPS_DIR}/kernel/cpu)
file(GLOB_RECURSE NNAPI_RUNTIME_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/op/*.cc
        )
add_library(nnapi_kernel_mid OBJECT ${NNAPI_RUNTIME_SRC})
add_dependencies(nnapi_kernel_mid fbs_src)
