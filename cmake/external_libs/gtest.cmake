set(gtest_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(gtest_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

set(CMAKE_OPTION
        -DBUILD_TESTING=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=ON
        -DCMAKE_MACOSX_RPATH=TRUE)

if(NOT ENABLE_GLIBCXX)
    set(gtest_CXXFLAGS "${gtest_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/googletest/repository/archive/release-1.12.1.tar.gz")
    set(SHA256 "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2")
else()
    set(REQ_URL "https://github.com/google/googletest/archive/release-1.12.1.tar.gz")
    set(SHA256 "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2")
endif()

mindspore_add_pkg(gtest
        VER 1.12.1
        LIBS gtest gmock
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION})

include_directories(${gtest_INC})
add_library(mindspore::gtest ALIAS gtest::gtest)
add_library(mindspore::gmock ALIAS gtest::gmock)
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    file(COPY ${gtest_DIRPATH}/bin/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_DIRPATH}/bin/libgtest_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_DIRPATH}/bin/libgmock_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_DIRPATH}/bin/libgmock${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
else()
    file(COPY ${gtest_LIBPATH}/libgtest${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_LIBPATH}/libgtest_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_LIBPATH}/libgmock${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
    file(COPY ${gtest_LIBPATH}/libgmock_main${CMAKE_SHARED_LIBRARY_SUFFIX} DESTINATION
            ${CMAKE_BINARY_DIR}/googletest/googlemock/gtest FOLLOW_SYMLINK_CHAIN)
endif()