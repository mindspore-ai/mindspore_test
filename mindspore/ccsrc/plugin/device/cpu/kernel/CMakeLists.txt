file(GLOB_RECURSE CPU_KERNEL_OBJECTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(BUILD_LITE)
    # mslite do not support python op
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx ")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -maxv ")
    string(REPLACE "-Wall" "" CMAKE_C_FLAGS ${CMAKE_C_FLAGS})
    string(REPLACE "-Wall" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
    list(REMOVE_ITEM CPU_KERNEL_OBJECTS "pyexecute/py_execute_cpu_kernel.cc")
    list(REMOVE_ITEM CPU_KERNEL_OBJECTS "pyfunc/py_func_cpu_kernel.cc")
    list(REMOVE_ITEM CPU_KERNEL_OBJECTS "opaque_predicate_kernel.cc")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux" AND ENABLE_CPU)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else()
    file(GLOB_RECURSE AKG_CPU_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "akg/*.cc"
        "dynamic_akg/*.cc"
    )
    list(REMOVE_ITEM CPU_KERNEL_OBJECTS ${AKG_CPU_SRC_LIST})
endif()

if(ENABLE_CPU)
    add_library(_mindspore_plugin_device_cpu_kernel_obj OBJECT ${CPU_KERNEL_OBJECTS})
else() # for ut
    add_library(_mindspore_plugin_device_cpu_kernel_obj OBJECT "cpu_kernel.cc")
endif()

