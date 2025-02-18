if(MSVC)
    set(tinyxml2_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    set(tinyxml2_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    if(DEBUG_MODE)
        set(tinyxml2_Debug ON)
    endif()
else()
    set(tinyxml2_CXXFLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -fPIC")
    set(tinyxml2_CFLAGS "-fstack-protector -D_FORTIFY_SOURCE=2 -O2 -fPIC")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/tinyxml2/repository/archive/10.0.0.tar.gz")
    set(SHA256 "3bdf15128ba16686e69bce256cc468e76c7b94ff2c7f391cc5ec09e40bff3839")
else()
    set(REQ_URL "https://github.com/leethomason/tinyxml2/archive/10.0.0.tar.gz")
    set(SHA256 "3bdf15128ba16686e69bce256cc468e76c7b94ff2c7f391cc5ec09e40bff3839")
endif()


if(NOT WIN32 AND NOT APPLE)
    set(tinyxml2_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

mindspore_add_pkg(tinyxml2
        VER 10.0.0
        LIBS tinyxml2
        URL ${REQ_URL}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON
        SHA256 ${SHA256})
include_directories(${tinyxml2_INC})
add_library(mindspore::tinyxml2 ALIAS tinyxml2::tinyxml2)
