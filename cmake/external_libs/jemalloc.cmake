if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/jemalloc.git")
    set(GIT_TAG "master")
    set(SHA256 "2db82d1e7119df3e71b7640219b6dfe84789bc0537983c3b7ac4f7189aecfeaa")
    set(JEMALLOC_SRC "${TOP_DIR}/build/mindspore/_deps/jemalloc-src")
    __download_pkg_with_git(jemalloc ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${JEMALLOC_SRC}/jemalloc-5.3.0.tar.bz2 --strip-components 1 -C ${JEMALLOC_SRC})
else()
    set(REQ_URL "https://github.com/jemalloc/jemalloc/archive/refs/tags/5.3.0.tar.gz")
    set(SHA256 "ef6f74fd45e95ee4ef7f9e19ebe5b075ca6b7fbe0140612b2a161abafb7ee179")
endif()
set(PRE_CONFIGURE_CMD "./autogen.sh")


if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("jemalloc thirdparty do not support windows currently.")
else()
    set(jemalloc_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all")
    set(jemalloc_CFLAGS "-D_FORTIFY_SOURCE=2 -O2 -fstack-protector-all")
    if(NOT ${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(jemalloc_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
    mindspore_add_pkg(jemalloc
            VER 5.3.0
            LIBS jemalloc
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PRE_CONFIGURE_COMMAND ${PRE_CONFIGURE_CMD}
            CONFIGURE_COMMAND ./configure)

    include_directories(${jemalloc_INC})
    add_library(mindspore::jemalloc ALIAS jemalloc::jemalloc)
endif()
