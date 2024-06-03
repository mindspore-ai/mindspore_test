# include(${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/gencode.cmake)
set(INFER_DIR "infer")
set(VIEW_DIR "view")
set(UTILS_DIR "ops_utils")

# ------- compiler flags ------
if("${ENABLE_HIDDEN}" STREQUAL "OFF" AND NOT MSVC)
    string(REPLACE " -Werror " " " CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    string(REPLACE " -fvisibility=hidden" " -fvisibility=default" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
endif()

if(NOT "${CMAKE_C_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()
if(NOT "${CMAKE_CXX_FLAGS}" STREQUAL "")
    string(REPLACE "-fno-rtti" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-fno-exceptions" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif()

# ------- CORE_OPS_LIST, CORE_SYMBOL_OPS_LIST ------
if(CMAKE_SIZEOF_VOID_P EQUAL 4 OR NOT BUILD_LITE)
    file(GLOB_RECURSE CORE_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/*.cc")
    file(GLOB_RECURSE CORE_SYMBOL_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/symbol_ops_impl/*.cc")
    file(GLOB_RECURSE VIEW_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${VIEW_DIR}/*.cc")
    file(GLOB_RECURSE UTILS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${UTILS_DIR}/*.cc")
    set_property(SOURCE ${CORE_SYMBOL_OPS_LIST} PROPERTY COMPILE_DEFINITIONS
                SUBMODULE_ID=mindspore::SubModuleId::SM_SYMBOLIC_SHAPE)
    list(APPEND CORE_OPS_LIST ${VIEW_LIST} ${UTILS_LIST} ${OP_DEF_SRC})
else()
    # ------- LITE merge_files -----
    include(${TOP_DIR}/mindspore/lite/cmake/merge.cmake)
    if(ENABLE_SECURITY)
        merge_files(${CMAKE_CURRENT_SOURCE_DIR}/infer/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops infer_merge
                    "_summary.cc$")
    else()
        merge_files(${CMAKE_CURRENT_SOURCE_DIR}/infer/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops infer_merge "")
    endif()
    merge_files(${CMAKE_CURRENT_SOURCE_DIR}/op_def/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops op_def_merge "")
    merge_files(${CMAKE_CURRENT_SOURCE_DIR}/view/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops view_merge "")
    merge_files(${CMAKE_CURRENT_SOURCE_DIR}/ops_utils/ ${CMAKE_BINARY_DIR}/merge/mindspore/ops utils_merge "")
    file(GLOB_RECURSE CORE_OPS_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_BINARY_DIR}/merge/mindspore/ops/*.cc")
endif()



# -------- CORE_SRC_LIST, SYMBOLIC_SHAPE_SRC_LIST ----------
if(TARGET_AOS_ARM)
    string(REPLACE "-Werror" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    string(REPLACE "-Werror" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
endif()

if(ANDROID_NDK)
    add_definitions(-w)
    add_compile_definitions(KERNEL_EXECUTOR_ANDROID)
    set(TARGET_AOS_ARM ON)
endif()

if(ENABLE_SECURITY)
    file(GLOB_RECURSE _INFER_SUMMARY_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "${INFER_DIR}/*_summary.cc")
    list(REMOVE_ITEM CORE_OPS_LIST ${_INFER_SUMMARY_FILES})
endif()
# ------------------ COMPILE_DEFINITIONS -------------------

if(NOT MSLITE_TARGET_SITEAI)
    set(MSLITE_DEPS_OPENSSL on)
    set(MSLITE_SIMPLEST_CLOUD_INFERENCE off)
endif()

if(MSLITE_DEPS_OPENSSL)
    add_compile_definitions(ENABLE_OPENSSL)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -DHAVE_SNPRINTF")
  endif()
  add_compile_definitions(OPS_DLL)
elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
    -Wno-overloaded-virtual -Wno-user-defined-warnings -Winconsistent-missing-override -Wno-delete-non-virtual-dtor \
    -Wno-unused-private-field")
endif()

# ------------------- COMPILE, ADD_TARGET ---------------------
set(OPS_OBJECT_COUNT 1)
src_separate_compile(
    OBJECT_NAME ops_obj
    OBJECT_SIZE OPS_OBJECT_COUNT
    SRC_LIST ${CORE_OPS_LIST})
foreach(number RANGE 1 ${OPS_OBJECT_COUNT})
    list(APPEND OPS_OBJECT_LIST $<TARGET_OBJECTS:ops_obj_${number}>)
endforeach()

set(OPS_OBJECT_COUNT "${OPS_OBJECT_COUNT}" PARENT_SCOPE)
add_library(mindspore_ops SHARED ${OPS_OBJECT_LIST})
add_dependencies(mindspore_ops generated_code)

# ------------------ LINK, SET_PROPERTY ---------------

target_link_libraries(mindspore_ops PRIVATE securec)

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_options(mindspore_ops PRIVATE -Wl,-init,mindspore_log_init)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set_target_properties(mindspore_ops PROPERTIES MACOSX_RPATH ON)
    set_target_properties(mindspore_ops PROPERTIES INSTALL_RPATH @loader_path)
else()
    set_target_properties(mindspore_ops PROPERTIES INSTALL_RPATH $ORIGIN)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_ops PRIVATE mindspore::glog)
endif()

if((${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE) AND (NOT TARGET_AOS_ARM) AND (NOT ANDROID_NDK) AND
        (NOT MSLITE_SIMPLEST_CLOUD_INFERENCE))
    target_link_libraries(mindspore_ops PRIVATE mindspore::crypto -pthread)
endif()

if(ANDROID_NDK)
    target_link_libraries(mindspore_ops PRIVATE -llog)
endif()

target_link_libraries(mindspore_ops PRIVATE mindspore_core)