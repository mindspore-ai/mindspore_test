find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    find_package(Python3 COMPONENTS NumPy Development)
    if(Python3_NumPy_FOUND)
        include_directories(${Python3_NumPy_INCLUDE_DIRS})
        message(STATUS "Python3_NumPy_INCLUDE_DIRS: ${Python3_NumPy_INCLUDE_DIRS}")
    else()
        message(FATAL_ERROR "Required package Numpy not found, "
                "please install Numpy<2.0 and try building MindSpore again.")
    endif()
endif()

file(GLOB_RECURSE _NP_DTYPES_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_NP_DTYPES_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
    SUBMODULE_ID=mindspore::SubModuleId::SM_ANALYZER)
add_library(_mindspore_frontend_np_dtypes_obj OBJECT ${_NP_DTYPES_SRC_FILES})
