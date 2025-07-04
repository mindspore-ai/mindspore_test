if(DEFINED ENV{ASCEND_CUSTOM_PATH})
    set(TOOLCHAIN_PATH $ENV{ASCEND_CUSTOM_PATH}/latest/toolkit/toolchain)
elseif(EXISTS /usr/local/Ascend/latest/toolkit/toolchain)
    set(TOOLCHAIN_PATH /usr/local/Ascend/latest/toolkit/toolchain)
elseif(EXISTS /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain)
    set(TOOLCHAIN_PATH /usr/local/Ascend/ascend-toolkit/latest/toolkit/toolchain)
else()
    set(TOOLCHAIN_PATH /usr/local/Ascend/toolkit/toolchain)
endif()
set(CMAKE_C_COMPILER ${TOOLCHAIN_PATH}/hcc/bin/aarch64-target-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PATH}/hcc/bin/aarch64-target-linux-gnu-g++)

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if(EXISTS ${CMAKE_C_COMPILER} AND EXISTS ${CMAKE_CXX_COMPILER})
    find_program(CCACHE_EXECUTABLE ccache)
    if(CCACHE_EXECUTABLE)
        message(STATUS "using ccache to compile aicpu kernels")
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
    endif()
    if(ENABLE_SYM_FILE)
        string(REPLACE "-ggdb" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
        string(REPLACE "-g" " " CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
    endif()

    include(${CMAKE_SOURCE_DIR}/cmake/dependency_securec_arm.cmake)
    include(${CMAKE_SOURCE_DIR}/cmake/external_libs/protobuf_arm.cmake)

    set(AICPU_PROTO_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_op_proto/aicpu_tensor.proto
    )

    ms_protobuf_generate(PROTO_SRCS PROTO_HDRS ${AICPU_PROTO_SRC})

    set(AICPU_SRC
        ${PROTO_SRCS}
        ${CMAKE_CURRENT_SOURCE_DIR}/common/kernel_base.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/range_sampler.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/common/random_utils.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cpu_kernel/utils/cust_op_log.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_sharder/aicpu_async_event.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_sharder/aicpu_context.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/aicpu_sharder/aicpu_pulse.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_buffer/fifo_replay_buffer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_buffer/priority_replay_buffer_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_buffer/priority_replay_buffer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_buffer/reservoir_replay_buffer_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/replay_buffer/reservoir_replay_buffer.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/candidate_sampler_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/cast_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/concat_offset_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/meshgrid_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/drop_out_gen_mask_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/expand_dims_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/fse_decode_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/gather_grad_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/poisson_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/quant_dtype_cast_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/range_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reshape_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence_add_offset.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence_add.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence_addn.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence_concat.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/sequence_stack.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/slice_grad_kernel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/squeeze_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/standard_normal_kernels.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/uniform_real_kernels.cc
    )

    string(REGEX REPLACE "-B.*mold" " " CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REGEX REPLACE "-B.*mold" " " CMAKE_C_FLAGS ${CMAKE_C_FLAGS})

    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/customize/op_info_cfg)
        add_subdirectory(customize)
    endif()
    add_subdirectory(cpu_kernel)
else()
    message("[WARNING]Due to the lack of cross compiler, the aicpu operator is not compiled.
    If you need to use a custom aicpu operator, please confirm whether the relevant directory
    [${CMAKE_C_COMPILER}] and [${CMAKE_CXX_COMPILER}] is exists")
endif()
