file(GLOB_RECURSE KERNEL_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "kernel_factory.cc"
    "kernel_build_info.cc"
    "kernel_info.cc"
    "ops_utils.cc"
    "common_utils.cc"
    "format_utils.cc"
    "framework_utils.cc"
    "philox_random.cc"
    "kash/*.cc"
    "oplib/*.cc"
    "environ_manager.cc"
    "kernel_mod_cache.cc"
)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-non-abstract-non-virtual-dtor -Wno-overloaded-virtual")
endif()

if(ENABLE_AKG AND ${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "graph_kernel/graph_kernel_json_generator.cc"
        "graph_kernel/graph_kernel_builder.cc"
        "graph_kernel/graph_kernel_builder_manager.cc"
        "graph_kernel/set_infershape_functor.cc"
        "graph_kernel/fake_abstract_shape.cc"
        "graph_kernel/akg/akg_kernel_build.cc"
        "graph_kernel/akg/akg_kernel_json_decoder.cc"
        "graph_kernel/dynamic_akg/dynamic_akg_kernel_build.cc"
        "graph_kernel/kernel_packet/*.cc"
        "graph_kernel/symbol_engine/*.cc"
        "graph_kernel/symbol_engine/jit/*.cc"
    )
    list(APPEND KERNEL_SRC_LIST "${AKG_SRC_LIST}")
else()
    file(GLOB_RECURSE AKG_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "graph_kernel/graph_kernel_json_generator.cc"
        "graph_kernel/akg/akg_kernel_json_decoder.cc"
        "graph_kernel/kernel_packet/*.cc"
        "graph_kernel/set_infershape_functor.cc"
        "graph_kernel/fake_abstract_shape.cc"
        "graph_kernel/symbol_engine/*.cc"
        "graph_kernel/symbol_engine/jit/*.cc"
    )
    list(APPEND KERNEL_SRC_LIST "${AKG_SRC_LIST}")
endif()

set_property(SOURCE ${KERNEL_SRC_LIST} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_KERNEL)
add_library(_mindspore_kernel_obj OBJECT ${KERNEL_SRC_LIST})
target_link_libraries(_mindspore_kernel_obj PRIVATE mindspore_core mindspore_ops)
