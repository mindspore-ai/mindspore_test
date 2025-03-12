set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(MSVC)
    set(cppjieba_CXXFLAGS "/utf-8")
    set(cppjieba_CFLAGS "/utf-8")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/cppjieba/repository/archive/v5.1.1.tar.gz")
    set(SHA256 "3ab42d945bb8dd313080267dcda1ecd7be2d02667c86a351c423478951c759b0")
else()
    set(REQ_URL "https://github.com/yanyiwu/cppjieba/archive/v5.1.1.tar.gz")
    set(SHA256 "88496758dd2ab495fe9a7cdcd7779f0688bc51304ae01467a0010817617a2a28")
endif()

set(limonp_name "limonp")
set(limonp_url "https://github.com/yanyiwu/limonp/archive/refs/tags/v0.6.6.tar.gz")
set(limonp_sha256 "3a69a673a5f12e83660f699c43b29fc4c509a2078aa901193264f40e94ca4d01")

function(download_limonp_package)
    set(LIMONP_SRC "${TOP_DIR}/build/mindspore/_deps/limonp-src")
    set(CPPJIEBA_SRC "${TOP_DIR}/build/mindspore/_deps/cppjieba-src")
    set(limonp_name ${limonp_name})
    set(limonp_url ${limonp_url})
    set(limonp_sha256 ${limonp_sha256})

    if(LOCAL_LIBS_SERVER)
        set(REGEX_IP_ADDRESS "^([0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+)$")
        get_filename_component(LIMONP_FILE_NAME ${limonp_url} NAME)
        if(${LOCAL_LIBS_SERVER} MATCHES ${REGEX_IP_ADDRESS})
            set(limonp_url "http://${LOCAL_LIBS_SERVER}:8081/libs/${limonp_name}/${LIMONP_FILE_NAME}"
                    ${limonp_url})
        else()
            set(limonp_url "https://${LOCAL_LIBS_SERVER}/libs/${limonp_name}/${LIMONP_FILE_NAME}" ${limonp_url})
        endif()
    endif()

    FetchContent_Declare(
            ${limonp_name}
            URL      ${limonp_url}
            URL_HASH SHA256=${limonp_sha256}
    )
    FetchContent_GetProperties(${limonp_name})

    if(NOT ${limonp_name}_POPULATED)
        FetchContent_Populate(${limonp_name})
        set(${limonp_name}_SOURCE_DIR ${${limonp_name}_SOURCE_DIR} PARENT_SCOPE)
    endif()

    file(COPY "${LIMONP_SRC}/." DESTINATION "${CPPJIEBA_SRC}/deps/limonp")
endfunction()

set(limonp_hash_context "${limonp_name}-${limonp_url}-${limonp_sha256}")

mindspore_add_pkg(cppjieba
        VER 5.1.1
        HEAD_ONLY ./
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/cppjieba/cppjieba.patch
        PATCHES ${TOP_DIR}/third_party/patch/cppjieba/cppjieba_msvc_compile.patch
        CUSTOM_SUBMODULE_DOWNLOAD download_limonp_package
        CUSTOM_SUBMODULE_INFO ${limonp_hash_context})

include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps/limonp/include)
add_library(mindspore::cppjieba ALIAS cppjieba)

