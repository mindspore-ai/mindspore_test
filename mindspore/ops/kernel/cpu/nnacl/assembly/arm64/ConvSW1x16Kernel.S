/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

// void SWConv1x16Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
//                       size_t kernel_w, size_t act_flag, size_t out_step, size_t ic_algin, size_t in_kw_step,
//                       size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode)
// x0: dst, x1: src, x2: weight, x3: bias, x4: kernel_h, x5: kernel_w, x6: act_flag, x7: out_step,
// x10: ic_algin, x11: in_kw_step, x12: in_kh_step, x13: in_sw_step, x14: kw_remainder, x15: write_mode
asm_function SWConv1x16Kernel
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #64
    stp x19, x20, [sp]
    stp x21, x22, [sp, #16]
    stp x23, x24, [sp, #32]
    stp x25, x26, [sp, #48]

    ldr x10, [sp, #64]
    ldr x11, [sp, #72]
    ldr x12, [sp, #80]
    ldr x13, [sp, #88]
    ldr x14, [sp, #96]
    ldr x15, [sp, #104]
    lsl x7, x7, #2
    lsl x11, x11, #2
    lsl x12, x12, #2
    lsl x13, x13, #2
    lsl x14, x14, #2
    add x20, x0, x7

    cbz x3, InitNoBias
    InitWithBias:
        ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x3]
        b LoopH
    InitNoBias:
        dup v0.2d, xzr
        dup v1.2d, xzr
        dup v2.2d, xzr
        dup v3.2d, xzr

    LoopH:
        mov x22, x5
        mov x23, x1
        LoopW:
            prfm pldl1keep, [x23]
            mov x24, x23
            mov x25, x10
            subs x25, x25, #16
            blt LoopC12
            LoopC16:
                ld1 {v4.4s, v5.4s, v6.4s, v7.4s}, [x24], #64
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[0]
                fmla v1.4s, v17.4s, v4.s[0]
                fmla v2.4s, v18.4s, v4.s[0]
                fmla v3.4s, v19.4s, v4.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[1]
                fmla v1.4s, v17.4s, v4.s[1]
                fmla v2.4s, v18.4s, v4.s[1]
                fmla v3.4s, v19.4s, v4.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[2]
                fmla v1.4s, v17.4s, v4.s[2]
                fmla v2.4s, v18.4s, v4.s[2]
                fmla v3.4s, v19.4s, v4.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[3]
                fmla v1.4s, v17.4s, v4.s[3]
                fmla v2.4s, v18.4s, v4.s[3]
                fmla v3.4s, v19.4s, v4.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[0]
                fmla v1.4s, v17.4s, v5.s[0]
                fmla v2.4s, v18.4s, v5.s[0]
                fmla v3.4s, v19.4s, v5.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[1]
                fmla v1.4s, v17.4s, v5.s[1]
                fmla v2.4s, v18.4s, v5.s[1]
                fmla v3.4s, v19.4s, v5.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[2]
                fmla v1.4s, v17.4s, v5.s[2]
                fmla v2.4s, v18.4s, v5.s[2]
                fmla v3.4s, v19.4s, v5.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[3]
                fmla v1.4s, v17.4s, v5.s[3]
                fmla v2.4s, v18.4s, v5.s[3]
                fmla v3.4s, v19.4s, v5.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[0]
                fmla v1.4s, v17.4s, v6.s[0]
                fmla v2.4s, v18.4s, v6.s[0]
                fmla v3.4s, v19.4s, v6.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[1]
                fmla v1.4s, v17.4s, v6.s[1]
                fmla v2.4s, v18.4s, v6.s[1]
                fmla v3.4s, v19.4s, v6.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[2]
                fmla v1.4s, v17.4s, v6.s[2]
                fmla v2.4s, v18.4s, v6.s[2]
                fmla v3.4s, v19.4s, v6.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[3]
                fmla v1.4s, v17.4s, v6.s[3]
                fmla v2.4s, v18.4s, v6.s[3]
                fmla v3.4s, v19.4s, v6.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v7.s[0]
                fmla v1.4s, v17.4s, v7.s[0]
                fmla v2.4s, v18.4s, v7.s[0]
                fmla v3.4s, v19.4s, v7.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v7.s[1]
                fmla v1.4s, v17.4s, v7.s[1]
                fmla v2.4s, v18.4s, v7.s[1]
                fmla v3.4s, v19.4s, v7.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v7.s[2]
                fmla v1.4s, v17.4s, v7.s[2]
                fmla v2.4s, v18.4s, v7.s[2]
                fmla v3.4s, v19.4s, v7.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v7.s[3]
                fmla v1.4s, v17.4s, v7.s[3]
                fmla v2.4s, v18.4s, v7.s[3]
                fmla v3.4s, v19.4s, v7.s[3]
                subs x25, x25, #16
                bge LoopC16
            LoopC12:
                adds x25, x25, #16
                cbz x25, LoopCEnd
                cmp x25, #12
                blt LoopC8
                ld1 {v4.4s, v5.4s, v6.4s}, [x24], #48
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[0]
                fmla v1.4s, v17.4s, v4.s[0]
                fmla v2.4s, v18.4s, v4.s[0]
                fmla v3.4s, v19.4s, v4.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[1]
                fmla v1.4s, v17.4s, v4.s[1]
                fmla v2.4s, v18.4s, v4.s[1]
                fmla v3.4s, v19.4s, v4.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[2]
                fmla v1.4s, v17.4s, v4.s[2]
                fmla v2.4s, v18.4s, v4.s[2]
                fmla v3.4s, v19.4s, v4.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[3]
                fmla v1.4s, v17.4s, v4.s[3]
                fmla v2.4s, v18.4s, v4.s[3]
                fmla v3.4s, v19.4s, v4.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[0]
                fmla v1.4s, v17.4s, v5.s[0]
                fmla v2.4s, v18.4s, v5.s[0]
                fmla v3.4s, v19.4s, v5.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[1]
                fmla v1.4s, v17.4s, v5.s[1]
                fmla v2.4s, v18.4s, v5.s[1]
                fmla v3.4s, v19.4s, v5.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[2]
                fmla v1.4s, v17.4s, v5.s[2]
                fmla v2.4s, v18.4s, v5.s[2]
                fmla v3.4s, v19.4s, v5.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[3]
                fmla v1.4s, v17.4s, v5.s[3]
                fmla v2.4s, v18.4s, v5.s[3]
                fmla v3.4s, v19.4s, v5.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[0]
                fmla v1.4s, v17.4s, v6.s[0]
                fmla v2.4s, v18.4s, v6.s[0]
                fmla v3.4s, v19.4s, v6.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[1]
                fmla v1.4s, v17.4s, v6.s[1]
                fmla v2.4s, v18.4s, v6.s[1]
                fmla v3.4s, v19.4s, v6.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[2]
                fmla v1.4s, v17.4s, v6.s[2]
                fmla v2.4s, v18.4s, v6.s[2]
                fmla v3.4s, v19.4s, v6.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.s[3]
                fmla v1.4s, v17.4s, v6.s[3]
                fmla v2.4s, v18.4s, v6.s[3]
                fmla v3.4s, v19.4s, v6.s[3]
                sub x25, x25, #12
                b LoopCTail
            LoopC8:
                cmp x25, #8
                blt LoopC4
                ld1 {v4.4s, v5.4s}, [x24], #32
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[0]
                fmla v1.4s, v17.4s, v4.s[0]
                fmla v2.4s, v18.4s, v4.s[0]
                fmla v3.4s, v19.4s, v4.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[1]
                fmla v1.4s, v17.4s, v4.s[1]
                fmla v2.4s, v18.4s, v4.s[1]
                fmla v3.4s, v19.4s, v4.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[2]
                fmla v1.4s, v17.4s, v4.s[2]
                fmla v2.4s, v18.4s, v4.s[2]
                fmla v3.4s, v19.4s, v4.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[3]
                fmla v1.4s, v17.4s, v4.s[3]
                fmla v2.4s, v18.4s, v4.s[3]
                fmla v3.4s, v19.4s, v4.s[3]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[0]
                fmla v1.4s, v17.4s, v5.s[0]
                fmla v2.4s, v18.4s, v5.s[0]
                fmla v3.4s, v19.4s, v5.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[1]
                fmla v1.4s, v17.4s, v5.s[1]
                fmla v2.4s, v18.4s, v5.s[1]
                fmla v3.4s, v19.4s, v5.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[2]
                fmla v1.4s, v17.4s, v5.s[2]
                fmla v2.4s, v18.4s, v5.s[2]
                fmla v3.4s, v19.4s, v5.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.s[3]
                fmla v1.4s, v17.4s, v5.s[3]
                fmla v2.4s, v18.4s, v5.s[3]
                fmla v3.4s, v19.4s, v5.s[3]
                sub x25, x25, #8
                b LoopCTail
            LoopC4:
                cmp x25, #4
                blt LoopCTail
                ld1 {v4.4s}, [x24], #16
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[0]
                fmla v1.4s, v17.4s, v4.s[0]
                fmla v2.4s, v18.4s, v4.s[0]
                fmla v3.4s, v19.4s, v4.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[1]
                fmla v1.4s, v17.4s, v4.s[1]
                fmla v2.4s, v18.4s, v4.s[1]
                fmla v3.4s, v19.4s, v4.s[1]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[2]
                fmla v1.4s, v17.4s, v4.s[2]
                fmla v2.4s, v18.4s, v4.s[2]
                fmla v3.4s, v19.4s, v4.s[2]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[3]
                fmla v1.4s, v17.4s, v4.s[3]
                fmla v2.4s, v18.4s, v4.s[3]
                fmla v3.4s, v19.4s, v4.s[3]
                sub x25, x25, #4
            LoopCTail:
                cbz x25, LoopCEnd
                cmp x25, #2
                beq LoopC2
                cmp x25, #1
                beq LoopC1
                // LoopC3
                ld3r {v4.4s, v5.4s, v6.4s}, [x24]
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.4s
                fmla v1.4s, v17.4s, v4.4s
                fmla v2.4s, v18.4s, v4.4s
                fmla v3.4s, v19.4s, v4.4s

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v5.4s
                fmla v1.4s, v17.4s, v5.4s
                fmla v2.4s, v18.4s, v5.4s
                fmla v3.4s, v19.4s, v5.4s

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v6.4s
                fmla v1.4s, v17.4s, v6.4s
                fmla v2.4s, v18.4s, v6.4s
                fmla v3.4s, v19.4s, v6.4s
                b LoopCEnd
            LoopC2:
                ld1 {v4.d}[0], [x24]
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[0]
                fmla v1.4s, v17.4s, v4.s[0]
                fmla v2.4s, v18.4s, v4.s[0]
                fmla v3.4s, v19.4s, v4.s[0]

                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.s[1]
                fmla v1.4s, v17.4s, v4.s[1]
                fmla v2.4s, v18.4s, v4.s[1]
                fmla v3.4s, v19.4s, v4.s[1]
                b LoopCEnd
            LoopC1:
                ld1r {v4.4s}, [x24]
                ld1 {v16.4s, v17.4s, v18.4s, v19.4s}, [x2], #64
                fmla v0.4s, v16.4s, v4.4s
                fmla v1.4s, v17.4s, v4.4s
                fmla v2.4s, v18.4s, v4.4s
                fmla v3.4s, v19.4s, v4.4s
            LoopCEnd:
                add x23, x23, x11
                subs x22, x22, #1
                bgt LoopW
        add x1, x1, x12
        add x2, x2, x14
        subs x4, x4, #1
        bgt LoopH

    ands x6, x6, #3
    beq WriteBack
    dup v4.2d, xzr          // relu
    fmax v0.4s, v0.4s, v4.4s
    fmax v1.4s, v1.4s, v4.4s
    fmax v2.4s, v2.4s, v4.4s
    fmax v3.4s, v3.4s, v4.4s

    ands x6, x6, #1
    beq WriteBack
    movi v4.4s, #6      // relu6
    scvtf v4.4s, v4.4s
    fmin v0.4s, v0.4s, v4.4s
    fmin v1.4s, v1.4s, v4.4s
    fmin v2.4s, v2.4s, v4.4s
    fmin v3.4s, v3.4s, v4.4s
    fmin v4.4s, v4.4s, v4.4s

    WriteBack:
        cmp x15, #13
        beq NC4HW4
        st1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0]
        b End
        NC4HW4:
            add x21, x0, x7, LSL #1
            add x22, x20, x7, LSL #1
            st1 {v0.4s}, [x0]
            st1 {v1.4s}, [x20]
            st1 {v2.4s}, [x21]
            st1 {v3.4s}, [x22]
    End:
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x25, x26, [sp], #16
    ret
#endif
