if(TARGET_AOS_ARM)
    set(CMAKE_C_COMPILER          "$ENV{CC}")
    set(CMAKE_SYSTEM_PROCESSOR    "aarch64")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/libjpeg-turbo/repository/archive/3.0.1.tar.gz")
    set(SHA256 "5b9bbca2b2a87c6632c821799438d358e27004ab528abf798533c15d50b39f82")
else()
    set(REQ_URL "https://github.com/libjpeg-turbo/libjpeg-turbo/archive/3.0.1.tar.gz")
    set(SHA256 "5b9bbca2b2a87c6632c821799438d358e27004ab528abf798533c15d50b39f82")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 \
        -O2")
else()
    if(MSVC)
        set(jpeg_turbo_CFLAGS "-O2")
    else()
        set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2")
        if(TARGET_AOS_ARM)
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -march=armv8.2-a -mtune=cortex-a72")
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -Wno-uninitialized -march=armv8.2-a -mtune=cortex-a72")
        else()
            set(jpeg_turbo_CFLAGS "${jpeg_turbo_CFLAGS} -Wno-maybe-uninitialized")
        endif()
    endif()
endif()

set(jpeg_turbo_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack,-s")


set(jpeg_turbo_USE_STATIC_LIBS ON)
set(CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_RPATH=TRUE -DWITH_SIMD=ON)

mindspore_add_pkg(jpeg_turbo
        VER 3.0.1
        LIBS jpeg turbojpeg
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION}
        )
include_directories(${jpeg_turbo_INC})
add_library(mindspore::jpeg_turbo ALIAS jpeg_turbo::jpeg)
add_library(mindspore::turbojpeg ALIAS jpeg_turbo::turbojpeg)
