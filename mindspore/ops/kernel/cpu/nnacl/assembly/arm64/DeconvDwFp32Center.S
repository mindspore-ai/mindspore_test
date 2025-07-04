/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifdef ENABLE_ARM64
#include "nnacl/assembly_global.h"

.text
.align 5

// void DeconvDwFp32Center(float *dst, const float *src, const float *weight, size_t height, size_t width,
//                      size_t kernel_h, size_t kernel_w, size_t out_h_step, size_t block_channel, size_t in_sh_step, size_t in_sw_step,
//                      size_t in_kh_step, size_t in_kw_step);
// x0: dst, x1: src, x2: weight, x3: height, x4: weight, x5: kernel_h, x6: kernel_w, x7: out_h_step
// x8: block_channel, x9: in_sh_step, x10: in_sw_step, x11: in_kh_step, x12: in_kw_step
asm_function DeconvDwFp32Center
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #32
    stp x19, x20, [sp]
    stp x21, x22, [sp, #16]

    ldr x8, [sp, #32]
    ldr x9, [sp, #40]
    ldr x10, [sp, #48]
    ldr x11, [sp, #56]
    ldr x12, [sp, #64]

    LoopH:
        mov x15, x0
        mov x16, x1
        mov x17, x4
        LoopW:
            mov x22, x15
            mov x19, x2
            mov x20, x5
            ld1 {v1.4s}, [x16], x8
            LoopKh:
                mov x21, x22
                mov x13, x6
                LoopKw:
                    ld1 {v0.4s}, [x21]
                    ld1 {v2.4s}, [x19], #16
                    fmla v0.4s, v1.4s, v2.4s
                    st1 {v0.4s}, [x21], x12
                    subs x13, x13, #1
                    bne LoopKw
                add x22, x22, x11
                subs x20, x20, #1
                bne LoopKh
            add x15, x15, x10
            subs x17, x17, #1
            bne LoopW
        add x0, x0, x9
        add x1, x1, x7
        subs x3, x3, #1
        bne LoopH

    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ret
#endif
