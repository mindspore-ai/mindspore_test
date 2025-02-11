set(cppjieba_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(cppjieba_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/cppjieba/repository/archive/v5.1.1.tar.gz")
    set(SHA256 "3ab42d945bb8dd313080267dcda1ecd7be2d02667c86a351c423478951c759b0")
else()
    set(REQ_URL "https://github.com/yanyiwu/cppjieba/archive/v5.1.1.tar.gz")
    set(SHA256 "88496758dd2ab495fe9a7cdcd7779f0688bc51304ae01467a0010817617a2a28")
endif()

mindspore_add_pkg(cppjieba
        VER 5.1.1
        HEAD_ONLY ./
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/cppjieba/cppjieba.patch)

include_directories(${cppjieba_INC}include)
include_directories(${cppjieba_INC}deps/limonp/include)
add_library(mindspore::cppjieba ALIAS cppjieba)

