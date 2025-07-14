set(securec_arm_USE_STATIC_LIBS ON)

# libboundscheck-v1.1.16
set(REQ_URL "https://gitee.com/openeuler/libboundscheck/repository/archive/v1.1.16.zip")
set(SHA256 "5119bda1ee96440c1a45e23f0cb8b079cc6697e052c4a78f27d0869f84ba312b")

string(REPLACE "/mindspore/lite" "" MS_TOP_DIR ${CMAKE_SOURCE_DIR})

mindspore_add_pkg(securec_arm
        VER 1.1.16
        LIBS securec
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION ${CMAKE_OPTION} -DTARGET_OHOS_LITE=OFF -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        PATCHES ${MS_TOP_DIR}/third_party/patch/securec/securec.patch001
        )

include_directories(${securec_arm_INC})
include_directories(${securec_arm_INC}/../)
add_library(mindspore::securec_arm ALIAS securec_arm::securec)