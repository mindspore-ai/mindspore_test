# the mindspore_extension_ascend.so will be added in the future.

# nnal is an optional package
if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    add_subdirectory(atb)
endif()

if(EXISTS ${ASCEND_NNAL_ASDSIP_PATH})
    add_definitions(-DASDSIP_PATH="${ASCEND_NNAL_ASDSIP_PATH}")
    add_subdirectory(asdsip)
endif()
