include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if("${ENABLE_HIDDEN}" STREQUAL "OFF")
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()


if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -Wno-user-defined-warnings -Wno-inconsistent-missing-override \
        -Wno-overloaded-virtual -Wno-unused-const-variable -Wno-pessimizing-move")
endif()

file(GLOB_RECURSE MS_LLM_BOOST_910B RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./boost_model_ascend_native.cc"
        )

set_property(SOURCE ${MS_LLM_BOOST_910B} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_LLM_BOOST)

if(ENABLE_D OR ENABLE_ACL)
        set(LCAL_LPATH ${ORIGIN_PATH}/ms_kernels_internal//asdops/lib)
        set(INTERNAL_LPATH ${ORIGIN_PATH}/ms_kernels_internal/internal_kernel/lib)
        include_directories($ENV{MS_INTERNAL_KERNEL_HOME})
        include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include)
        include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
        include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel)

        find_library(ASCENDCL ascendcl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
        find_library(MS_KERNELS_LIB ms_kernels_internal $ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/lib)
        find_library(LLAMA_OP llama_op $ENV{MS_INTERNAL_KERNEL_HOME}/internal_kernel/lib)
        find_library(LCAL_LIB lcal $ENV{MS_INTERNAL_KERNEL_HOME}/asdops/lib)

        add_library(ms_ascend_native_boost SHARED ${MS_LLM_BOOST_910B})
        target_link_libraries(ms_ascend_native_boost PRIVATE mindspore_backend mindspore_core mindspore_common)
        target_link_libraries(ms_ascend_native_boost PRIVATE securec d_collective)
        target_link_libraries(ms_ascend_native_boost PRIVATE ${ASCENDCL})
        target_link_libraries(ms_ascend_native_boost PRIVATE ${LLAMA_OP} ${MS_KERNELS_LIB} ${LCAL_LIB})
        set_target_properties(ms_ascend_native_boost PROPERTIES INSTALL_RPATH
                              ${ORIGIN_PATH}:${ORIGIN_PATH}/ascend:${ORIGIN_PATH}/../:${INTERNAL_LPATH}:${LCAL_LPATH})

endif()
