set(ASCEND_COLLECTIVE_SRCS "ascend_collective_comm_lib.cc"
        "ascend_communication_group.cc"
        "dummy_ascend_collective_comm_lib.cc"
        "ccool_collective_comm_lib.cc"
        "ccool_communication_group.cc"
        "leaper_trans.cc"
        "hccl_watch_dog_thread.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/dummy_collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
set_property(SOURCE ${ASCEND_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(d_collective STATIC ${ASCEND_COLLECTIVE_SRCS})
find_library(HCCL hccl ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
target_link_libraries(d_collective PRIVATE ${HCCL})

if(DEFINED ENV{MS_INTERNAL_KERNEL_HOME})
    include_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/include/lcal)
    link_directories($ENV{MS_INTERNAL_KERNEL_HOME}/asdops/lib)
    set(LOWLATENCY_COLLECTIVE_SRCS "lowlatency_collective_comm_lib.cc"
            "lowlatency_communication_group.cc"
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
            ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
    set_property(SOURCE ${LOWLATENCY_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
    add_library(lowlatency_collective SHARED ${LOWLATENCY_COLLECTIVE_SRCS})
    target_link_libraries(lowlatency_collective PRIVATE lcal mindspore_core)
    add_compile_definitions(ENABLE_INTERNAL_KERNELS)
endif()

set(DVM_KERNEL_HOME ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/plugin/device/ascend/kernel/dvm)
include_directories(${DVM_KERNEL_HOME})
link_directories(${DVM_LIB})
set(DVM_COLLECTIVE_SRCS "dvm_collective_comm_lib.cc"
        "dvm_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
set_property(SOURCE ${DVM_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(dvm_collective STATIC ${DVM_COLLECTIVE_SRCS})
target_link_libraries(dvm_collective PRIVATE ${DVM_LIB})

set(MULTI_ASCEND_COLLECTIVE_SRCS "multi_ascend_collective_comm_lib.cc"
        "multi_ascend_communication_group.cc"
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_communication_lib.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/collective_comm_lib_loader.cc
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/collective/communication_group.cc)
set_property(SOURCE ${MULTI_ASCEND_COLLECTIVE_SRCS} PROPERTY COMPILE_DEFINITIONS
        SUBMODULE_ID=mindspore::SubModuleId::SM_DEVICE)
add_library(multi_ascend_collective STATIC ${MULTI_ASCEND_COLLECTIVE_SRCS})
target_link_libraries(multi_ascend_collective PRIVATE d_collective PRIVATE dvm_collective)
