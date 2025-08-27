file(GLOB_RECURSE KERNEL_PACKET_SRC RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

add_library(_mindspore_ops_kernel_ascend_kernel_packet_obj OBJECT ${KERNEL_PACKET_SRC})
