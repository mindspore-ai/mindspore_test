# the mindspore_extension_ascend.so will be added in the future.

# nnal is an optional package
if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    add_subdirectory(atb)
endif()
