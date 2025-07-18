# default build aclnn
add_subdirectory(aclnn)

# nnal is an optional package
if(EXISTS ${ASCEND_NNAL_ATB_PATH})
    add_subdirectory(atb)
endif()
