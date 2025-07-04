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

// MatrixMultiplyWinogradFp16(float16_t *matix_a, float16_t *matrix_b, float16_t *matrix_c, int m, int k, int n, int in_channel)
    // x0: matrix_a, x1: matrix_b, x2: matrix_c, x3: m, x4: k, x5: n, x6: in_channel
asm_function MatrixMultiplyWinogradFp16
    // registers v8 ~ v15 must be preserved by a callee across subroutine calls, according to
    // https://github.com/ARM-software/abi-aa/blob/master/aapcs64/aapcs64.rst#simd-and-floating-point-registers
    // x19 ~ x29 should be also preserved
    // whereas our coding style do not permit such amount of parameters
    sub sp, sp, #48
    st1 {v8.8h}, [sp]
    stp x19, x20, [sp, #16]
    stp x21, x22, [sp, #32]

    mov x8, #2
    mul x10, x5, x8    // n * 2
    mov x17, x3  // m
    mul x13, x6, x8   // in_channel * 2
    mul x21, x13, x4  // in_channel * k * 2

    LoopM:
        mov x15, x5 // n
        mov x14, x1  // mat_b
        LoopN:
            mov x16, x0  // mat_a_m
            sub x22, x5, x15   // ni
            sub x19, x17, x3   // mi
            mul x22, x22, x17  // ni * m
            mov x11, x6 // in_channel
            add x22, x22, x19  // (ni * m) + mi
            mul x22, x22, x13   // x22 * channel_in * 2
            add x20, x2, x22   // dst + offset
            cmp x11, #32
            bge LoopC32
            cmp x11, #16
            bge LoopC16
            cmp x11, #8
            bge LoopC8
            cmp x11, #4
            bge LoopC4
            cmp x11, #1
            bge LoopC
            b EndLoopC
            LoopC32:
                mov x12, x14
                mov x9, x4  // new_k
                dup v5.8h, wzr
                dup v6.8h, wzr
                dup v7.8h, wzr
                dup v8.8h, wzr
                LoopK32:
                    ld1 {v0.8h, v1.8h, v2.8h, v3.8h}, [x16], x13
                    ldr h4, [x12]
                    add x12, x12, x10
                    fmla v5.8h, v0.8h, v4.h[0]
                    fmla v6.8h, v1.8h, v4.h[0]
                    fmla v7.8h, v2.8h, v4.h[0]
                    fmla v8.8h, v3.8h, v4.h[0]
                    subs x9, x9, #1
                    bne LoopK32
                Write32:
                    st1 {v5.8h}, [x20], #16
                    st1 {v6.8h}, [x20], #16
                    st1 {v7.8h}, [x20], #16
                    st1 {v8.8h}, [x20], #16

                sub x16, x16, x21  // back x13 * k
                add x16, x16, #64 // add 64B
                subs x11, x11, #32
                beq EndLoopC
                cmp x11, #32
                bge LoopC32
                cmp x11, #16
                bge LoopC16
                cmp x11, #8
                bge LoopC8
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC16:
                dup v5.8h, wzr
                dup v6.8h, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK16:
                    ld1 {v0.8h, v1.8h}, [x16], x13
                    ldr h4, [x12]
                    add x12, x12, x10
                    fmla v5.8h, v0.8h, v4.h[0]
                    fmla v6.8h, v1.8h, v4.h[0]
                    subs x9, x9, #1
                    bne LoopK16
                Write16:
                    st1 {v5.8h}, [x20], #16
                    st1 {v6.8h}, [x20], #16

                sub x16, x16, x21  // back x13 * k
                add x16, x16, #32 // add 32B
                subs x11, x11, #16
                beq EndLoopC
                cmp x11, #16
                bge LoopC16
                cmp x11, #8
                bge LoopC8
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC8:
                dup v5.8h, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK8:
                    ld1 {v0.8h}, [x16], x13
                    ldr h4, [x12]
                    add x12, x12, x10
                    fmla v5.8h, v0.8h, v4.h[0]
                    subs x9, x9, #1
                    bne LoopK8
                Write8:
                    st1 {v5.8h}, [x20], #16

                sub x16, x16, x21  // ptr back x13 * k
                add x16, x16, #16 // add 16B
                subs x11, x11, #8
                beq EndLoopC
                cmp x11, #8
                bge LoopC8
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC4:
                dup v5.4h, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK4:
                    ld1 {v0.4h}, [x16], x13
                    ldr h4, [x12]
                    add x12, x12, x10
                    fmla v5.4h, v0.4h, v4.h[0]
                    subs x9, x9, #1
                    bne LoopK4
                Write4:
                    st1 {v5.4h}, [x20], #8

                sub x16, x16, x21  // ptr back x13 * k
                add x16, x16, #8 // add 8B
                subs x11, x11, #4
                beq EndLoopC
                cmp x11, #4
                bge LoopC4
                cmp x11, #1
                bge LoopC

            LoopC:
                dup v5.8h, wzr
                mov x9, x4  // new_k
                mov x12, x14
                LoopK:
                    ldr h0, [x16]
                    add x16, x16, x13
                    ldr h4, [x12]
                    add x12, x12, x10
                    fmul h0, h0, h4
                    fadd h5, h5, h0
                    subs x9, x9, #1
                    bne LoopK
                Write:
                    str h5, [x20], #2

                sub x16, x16, x21  // ptr back x13 * k
                add x16, x16, #2 // ptr add 2B
                subs x11, x11, #1
                beq EndLoopC
                b LoopC

            EndLoopC:
                add x14, x14, #2
                subs x15, x15, #1
                beq EndLoopN
                b LoopN
        EndLoopN:
            subs x3, x3, #1
            beq EndLoopM
            add x0, x0, x21
            b LoopM

    EndLoopM:
        ld1 {v8.8h}, [sp], #16
        ldp x19, x20, [sp], #16
        ldp x21, x22, [sp], #16
    ret
#endif
