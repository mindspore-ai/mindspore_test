file(GLOB_RECURSE CPU_KERNEL_OBJECTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
if(ENABLE_CPU)
    add_library(_mindspore_plugin_device_cpu_kernel_obj OBJECT ${CPU_KERNEL_OBJECTS})
else() # for ut
    add_library(_mindspore_plugin_device_cpu_kernel_obj OBJECT "cpu_kernel.cc")
endif()

