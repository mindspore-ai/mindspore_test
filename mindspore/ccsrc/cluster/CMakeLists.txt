file(GLOB_RECURSE _CLUSTER_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")

if(NOT ENABLE_CPU OR WIN32 OR APPLE)
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/cluster_context.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/actor_route_table_proxy.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/compute_graph_node.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/meta_server_node.cc")
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/tcp_node.cc")
else()
    list(REMOVE_ITEM _CLUSTER_SRC_FILES "topology/dummy_cluster_context.cc")
endif()


if(NOT ENABLE_CPU OR WIN32 OR APPLE)
    set(EXCLUDE_DIRS "rpc/")
    foreach(EXCLUDE_DIR ${EXCLUDE_DIRS})
        foreach(EXCLUDE_FILE ${_CLUSTER_SRC_FILES})
            string(FIND ${EXCLUDE_FILE} ${EXCLUDE_DIR} FOUND)
            if(${FOUND} EQUAL 0)
                list(REMOVE_ITEM _CLUSTER_SRC_FILES ${EXCLUDE_FILE})
            endif()
        endforeach()
    endforeach()
endif()

set_property(SOURCE ${_CLUSTER_SRC_FILES} PROPERTY COMPILE_DEFINITIONS
            SUBMODULE_ID=mindspore::SubModuleId::SM_DISTRIBUTED)
add_library(_mindspore_cluster_obj OBJECT ${_CLUSTER_SRC_FILES})
