# include dependency
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# prepare output directory
file(REMOVE_RECURSE ${CMAKE_SOURCE_DIR}/output)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

# cpack variables
string(TOLOWER linux_${CMAKE_HOST_SYSTEM_PROCESSOR} PLATFORM_NAME)
set(CPACK_PACKAGE_FILE_NAME mindspore_ascend-${VERSION_NUMBER}-${PLATFORM_NAME})
set(CPACK_GENERATOR "TGZ")
set(CPACK_PACKAGE_CHECKSUM SHA256)
set(CPACK_PACKAGE_DIRECTORY ${CMAKE_SOURCE_DIR}/output)

set(INSTALL_LIB_DIR ${CMAKE_INSTALL_LIBDIR} CACHE PATH "Installation directory for libraries")
set(INSTALL_BASE_DIR ".")
set(INSTALL_BIN_DIR "utils/bin")
set(INSTALL_CFG_DIR "config")
set(INSTALL_LIB_DIR "lib")
set(INSTALL_PLUGIN_DIR "${INSTALL_LIB_DIR}/plugin")

# set package files
install(
        TARGETS mindspore_core mindspore_ops mindspore_common mindspore_ms_backend mindspore_pyboost mindspore_pynative
            mindspore_backend_manager mindspore_res_manager mindspore_frontend mindspore_profiler mindspore_memory_pool
            mindspore_runtime_pipeline mindspore_dump mindspore_backend_common mindspore_extension
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
)

if(ENABLE_CPU)
    install(
        TARGETS mindspore_ops_host LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
        NAMELINK_SKIP
    )
endif()

if(ENABLE_D)
    install(
        TARGETS mindspore_ge_backend
        DESTINATION ${INSTALL_LIB_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_ascend_res_manager
        DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_ascend
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
    )
endif()

if(ENABLE_GPU)
    install(
        TARGETS mindspore_gpu
        DESTINATION ${INSTALL_PLUGIN_DIR}
        COMPONENT mindspore
    )
    install(
        TARGETS mindspore_gpu_res_manager LIBRARY
        DESTINATION ${INSTALL_PLUGIN_DIR}/gpu
        COMPONENT mindspore
        NAMELINK_SKIP
    )
endif()

if(USE_GLOG)
    file(GLOB_RECURSE GLOG_LIB_LIST ${glog_LIBPATH}/libmindspore_glog*)
    install(
            FILES ${GLOG_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
endif()

if(ENABLE_MINDDATA)
    install(
            TARGETS _c_dataengine _c_mindrecord
            DESTINATION ${INSTALL_BASE_DIR}
            COMPONENT mindspore
    )
    install(
            TARGETS dataset-cache dataset-cache-server
            OPTIONAL
            DESTINATION ${INSTALL_BIN_DIR}
            COMPONENT mindspore
    )
    if(ENABLE_FFMPEG)
        file(GLOB_RECURSE FFMPEG_LIB_LIST
                ${ffmpeg_LIBPATH}/libavcodec*
                ${ffmpeg_LIBPATH}/libavdevice*
                ${ffmpeg_LIBPATH}/libavfilter*
                ${ffmpeg_LIBPATH}/libavformat*
                ${ffmpeg_LIBPATH}/libavutil*
                ${ffmpeg_LIBPATH}/libswresample*
                ${ffmpeg_LIBPATH}/libswscale*
                )
        install(
                FILES ${FFMPEG_LIB_LIST}
                DESTINATION ${INSTALL_LIB_DIR}
                COMPONENT mindspore
        )
    endif()
    file(GLOB_RECURSE OPENCV_LIB_LIST
            ${opencv_LIBPATH}/libopencv_core*
            ${opencv_LIBPATH}/libopencv_imgcodecs*
            ${opencv_LIBPATH}/libopencv_imgproc*
            )
    install(
            FILES ${OPENCV_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
    file(GLOB_RECURSE JEMALLOC_LIB_LIST
            ${jemalloc_LIBPATH}/libjemalloc*
            )
    install(
            FILES ${JEMALLOC_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
    file(GLOB_RECURSE TINYXML2_LIB_LIST ${tinyxml2_LIBPATH}/libtinyxml2*)
    install(
            FILES ${TINYXML2_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
    file(GLOB_RECURSE ICU4C_LIB_LIST
            ${icu4c_LIBPATH}/libicuuc*
            ${icu4c_LIBPATH}/libicudata*
            ${icu4c_LIBPATH}/libicui18n*
            )
    install(
            FILES ${ICU4C_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )

    if(ENABLE_ACL)
        install(
                TARGETS dvpp_utils
                DESTINATION ${INSTALL_PLUGIN_DIR}/ascend
                COMPONENT mindspore
        )
    endif()
endif()

# CPU mode
if(ENABLE_CPU AND NOT WIN32)
    install(
            TARGETS ps_cache
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
    install(
            TARGETS mindspore_cpu_res_manager
            DESTINATION ${INSTALL_PLUGIN_DIR}/cpu
            COMPONENT mindspore
)
endif()

if(ENABLE_CPU)
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/libdnnl*${CMAKE_SHARED_LIBRARY_SUFFIX}*)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
        file(GLOB_RECURSE DNNL_LIB_LIST ${onednn_LIBPATH}/dnnl.dll)
    endif()
    install(
            FILES ${DNNL_LIB_LIST}
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
    install(
            TARGETS nnacl
            DESTINATION ${INSTALL_LIB_DIR}
            COMPONENT mindspore
    )
endif()

if(MS_BUILD_GRPC)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc++.so.1.36.1
            DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc++.so.1 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_grpc.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_grpc.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_gpr.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_gpr.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_upb.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_upb.so.15 COMPONENT mindspore)
    install(FILES ${grpc_LIBPATH}/libmindspore_address_sorting.so.15.0.0
            DESTINATION ${INSTALL_LIB_DIR} RENAME libmindspore_address_sorting.so.15 COMPONENT mindspore)
endif()

## Public header files
install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/include
        DESTINATION ${INSTALL_BASE_DIR}
        COMPONENT mindspore
)

## Public header files for mindapi
install(
        DIRECTORY ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/base
        ${CMAKE_SOURCE_DIR}/mindspore/core/include/mindapi/ir
        DESTINATION ${INSTALL_BASE_DIR}/include/mindapi
        COMPONENT mindspore
)

## Public header files for minddata
install(
        FILES ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/config.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/constants.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/execute.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/text.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/transforms.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_lite.h
        ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include/dataset/vision_ascend.h
        DESTINATION ${INSTALL_BASE_DIR}/include/dataset
        COMPONENT mindspore
)

## config files
install(
        FILES ${CMAKE_SOURCE_DIR}/config/op_info.config
        DESTINATION ${INSTALL_CFG_DIR}
        COMPONENT mindspore
)

include(CPack)
