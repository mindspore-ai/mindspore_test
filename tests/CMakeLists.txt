#add flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")

if(ENABLE_TESTCASES)
    add_subdirectory(ut)
endif()
