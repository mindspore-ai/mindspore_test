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

// void SWConv2x8Kernel(float *dst, const float *src, const float *weight, const float *bias, size_t kernel_h,
//                      size_t kernel_w, size_t act_flag, size_t out_step, size_t ic_algin, size_t in_kw_step,
//                      size_t in_kh_step, size_t in_sw_step, size_t kw_remainder, size_t write_mode)
// x0: dst, x1: src, x2: weight, x3: bias, x4: kernel_h, x5: kernel_w, x6: act_flag, x7: out_step,
// x10: ic_algin, x11: in_kw_step, x12: in_kh_step, x13: in_sw_step, x14: kw_remainder, x15: write_mode
asm_function SWConv2x8Kernel
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
        ld1 {v0.4s, v1.4s}, [x3]
        ld1 {v2.4s, v3.4s}, [x3]
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
            mov x25, x10
            prfm pldl1keep, [x23]
            mov x24, x23
            add x26, x23, x13
            prfm pldl1keep, [x26]
            subs x25, x25, #12
            blt LoopC8
            LoopC12:
                ld1 {v16.4s, v17.4s, v18.4s}, [x24], #48
                ld1 {v19.4s, v20.4s, v21.4s}, [x26], #48
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[0]
                fmla v1.4s, v29.4s, v16.s[0]
                fmla v2.4s, v28.4s, v19.s[0]
                fmla v3.4s, v29.4s, v19.s[0]
                fmla v0.4s, v30.4s, v16.s[1]
                fmla v1.4s, v31.4s, v16.s[1]
                fmla v2.4s, v30.4s, v19.s[1]
                fmla v3.4s, v31.4s, v19.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[2]
                fmla v1.4s, v29.4s, v16.s[2]
                fmla v2.4s, v28.4s, v19.s[2]
                fmla v3.4s, v29.4s, v19.s[2]
                fmla v0.4s, v30.4s, v16.s[3]
                fmla v1.4s, v31.4s, v16.s[3]
                fmla v2.4s, v30.4s, v19.s[3]
                fmla v3.4s, v31.4s, v19.s[3]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v17.s[0]
                fmla v1.4s, v29.4s, v17.s[0]
                fmla v2.4s, v28.4s, v20.s[0]
                fmla v3.4s, v29.4s, v20.s[0]
                fmla v0.4s, v30.4s, v17.s[1]
                fmla v1.4s, v31.4s, v17.s[1]
                fmla v2.4s, v30.4s, v20.s[1]
                fmla v3.4s, v31.4s, v20.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v17.s[2]
                fmla v1.4s, v29.4s, v17.s[2]
                fmla v2.4s, v28.4s, v20.s[2]
                fmla v3.4s, v29.4s, v20.s[2]
                fmla v0.4s, v30.4s, v17.s[3]
                fmla v1.4s, v31.4s, v17.s[3]
                fmla v2.4s, v30.4s, v20.s[3]
                fmla v3.4s, v31.4s, v20.s[3]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v18.s[0]
                fmla v1.4s, v29.4s, v18.s[0]
                fmla v2.4s, v28.4s, v21.s[0]
                fmla v3.4s, v29.4s, v21.s[0]
                fmla v0.4s, v30.4s, v18.s[1]
                fmla v1.4s, v31.4s, v18.s[1]
                fmla v2.4s, v30.4s, v21.s[1]
                fmla v3.4s, v31.4s, v21.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v18.s[2]
                fmla v1.4s, v29.4s, v18.s[2]
                fmla v2.4s, v28.4s, v21.s[2]
                fmla v3.4s, v29.4s, v21.s[2]
                fmla v0.4s, v30.4s, v18.s[3]
                fmla v1.4s, v31.4s, v18.s[3]
                fmla v2.4s, v30.4s, v21.s[3]
                fmla v3.4s, v31.4s, v21.s[3]
                subs x25, x25, #12
                bge LoopC12
            LoopC8:
                adds x25, x25, #12
                cbz x25, LoopCEnd
                cmp x25, #8
                blt LoopC4
                ld1 {v16.4s, v17.4s}, [x24], #32
                ld1 {v19.4s, v20.4s}, [x26], #32
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[0]
                fmla v1.4s, v29.4s, v16.s[0]
                fmla v2.4s, v28.4s, v19.s[0]
                fmla v3.4s, v29.4s, v19.s[0]
                fmla v0.4s, v30.4s, v16.s[1]
                fmla v1.4s, v31.4s, v16.s[1]
                fmla v2.4s, v30.4s, v19.s[1]
                fmla v3.4s, v31.4s, v19.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[2]
                fmla v1.4s, v29.4s, v16.s[2]
                fmla v2.4s, v28.4s, v19.s[2]
                fmla v3.4s, v29.4s, v19.s[2]
                fmla v0.4s, v30.4s, v16.s[3]
                fmla v1.4s, v31.4s, v16.s[3]
                fmla v2.4s, v30.4s, v19.s[3]
                fmla v3.4s, v31.4s, v19.s[3]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v17.s[0]
                fmla v1.4s, v29.4s, v17.s[0]
                fmla v2.4s, v28.4s, v20.s[0]
                fmla v3.4s, v29.4s, v20.s[0]
                fmla v0.4s, v30.4s, v17.s[1]
                fmla v1.4s, v31.4s, v17.s[1]
                fmla v2.4s, v30.4s, v20.s[1]
                fmla v3.4s, v31.4s, v20.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v17.s[2]
                fmla v1.4s, v29.4s, v17.s[2]
                fmla v2.4s, v28.4s, v20.s[2]
                fmla v3.4s, v29.4s, v20.s[2]
                fmla v0.4s, v30.4s, v17.s[3]
                fmla v1.4s, v31.4s, v17.s[3]
                fmla v2.4s, v30.4s, v20.s[3]
                fmla v3.4s, v31.4s, v20.s[3]
                sub x25, x25, #8
                b LoopCTail
            LoopC4:
                cmp x25, #4
                blt LoopCTail
                ld1 {v16.4s}, [x24], #16
                ld1 {v19.4s}, [x26], #16
                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[0]
                fmla v1.4s, v29.4s, v16.s[0]
                fmla v2.4s, v28.4s, v19.s[0]
                fmla v3.4s, v29.4s, v19.s[0]
                fmla v0.4s, v30.4s, v16.s[1]
                fmla v1.4s, v31.4s, v16.s[1]
                fmla v2.4s, v30.4s, v19.s[1]
                fmla v3.4s, v31.4s, v19.s[1]

                ld1 {v28.4s, v29.4s, v30.4s, v31.4s}, [x2], #64
                fmla v0.4s, v28.4s, v16.s[2]
                fmla v1.4s, v29.4s, v16.s[2]
                fmla v2.4s, v28.4s, v19.s[2]
                fmla v3.4s, v29.4s, v19.s[2]
                fmla v0.4s, v30.4s, v16.s[3]
                fmla v1.4s, v31.4s, v16.s[3]
                fmla v2.4s, v30.4s, v19.s[3]
                fmla v3.4s, v31.4s, v19.s[3]
                sub x25, x25, #4
            LoopCTail:
                cbz x25, LoopCEnd
                LoopCTailCycle:
                    ld1 {v16.s}[0], [x24], #4
                    ld1 {v19.s}[0], [x26], #4
                    ld1 {v28.4s, v29.4s}, [x2], #32
                    fmla v0.4s, v28.4s, v16.s[0]
                    fmla v1.4s, v29.4s, v16.s[0]
                    fmla v2.4s, v28.4s, v19.s[0]
                    fmla v3.4s, v29.4s, v19.s[0]
                    subs x25, x25, #1
                    bgt LoopCTailCycle
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
    dup v24.2d, xzr          // relu
    fmax v0.4s, v0.4s, v24.4s
    fmax v1.4s, v1.4s, v24.4s
    fmax v2.4s, v2.4s, v24.4s
    fmax v3.4s, v3.4s, v24.4s

    ands x6, x6, #1
    beq WriteBack
    movi v24.4s, #6      // relu6
    scvtf v24.4s, v24.4s
    fmin v0.4s, v0.4s, v24.4s
    fmin v1.4s, v1.4s, v24.4s
    fmin v2.4s, v2.4s, v24.4s
    fmin v3.4s, v3.4s, v24.4s

    WriteBack:
        cmp x15, #13
        beq NC4HW4
        st1 {v0.4s, v1.4s}, [x0]
        st1 {v2.4s, v3.4s}, [x20]
        b End
        NC4HW4:
            st1 {v0.4s}, [x0], #16
            st1 {v2.4s}, [x0]
            st1 {v1.4s}, [x20], #16
            st1 {v3.4s}, [x20]
    End:
    ldp x19, x20, [sp], #16
    ldp x21, x22, [sp], #16
    ldp x23, x24, [sp], #16
    ldp x25, x26, [sp], #16
    ret
#endif
