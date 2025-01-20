if(ENABLE_D OR ENABLE_ACL)
    include(${CMAKE_SOURCE_DIR}/cmake/graphengine_variables.cmake)
    # include_directories(${CMAKE_CURRENT_SOURCE_DIR})

    # file(GLOB_RECURSE _GE_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    file(GLOB _GE_BACKEND_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
    set_property(SOURCE ${_GE_BACKEND_SRC_LIST}
          PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
    add_library(_mindspore_backend_ge_backend_obj OBJECT ${_GE_BACKEND_SRC_LIST})
    # add_library(mindspore_ge_backend SHARED ${_GE_BACKEND_SRC_LIST})

    add_subdirectory(pass)
endif()
