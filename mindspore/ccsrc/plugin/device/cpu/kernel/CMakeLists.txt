file(GLOB_RECURSE CPU_KERNEL_OBJECTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

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

