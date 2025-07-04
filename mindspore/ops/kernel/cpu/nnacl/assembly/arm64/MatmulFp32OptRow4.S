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

// void MatmulFloatNeon64OptRow4(const float *a, const float *b, float *c, const float *bias, int act_type, int depth
//                        int row, int col, size_t stride, size_t writeMode)
// x0: a
// x1: b
// x2: c
// x3: bias
// x4: act_type
// x5: depth
// x6: row
// x7: col
// x8: stride
// x9: writeMode

asm_function MatmulFloatNeon64OptRow4
    sub sp, sp, #160
    st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp]
    add x9, sp, #64
    st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x9]
    stp x19, x20, [sp, #128]
    stp x21, x22, [sp, #144]

    ldr x8, [sp, #160]
    ldr x9, [sp, #168]

    mov x21, #48 // sizeof(float) * 12

    mul x17, x5, x21 // block stride of lhs/rhs: sizeof(float) * 12 * depth
    cmp x9, #3 // c4
    beq C4Stride
    cbnz x9, NoC8Steps
    mov x11, x2
    mov x21, #32
    mul x16, x6, x21 // row * 8 * sizeof(float)
    b NoC8Steps
C4Stride:
    mov x18, #16 // 4 * sizeof(float)
    mov x22, #4
    mul x8, x8, x22 // stride * sizeof(float), in c4 stride == row
    mul x8, x8, x22 // col stride
    // col >= 4 , block stride 64, otherwise 4 * 4 * col
    cmp x7, #4
    bge C4StrideCommon
    mul x18, x18, x7 // block stride
    b LoopRowStart
C4StrideCommon:
    mov x18, #64 // block stride
    b LoopRowStart

NoC8Steps:
    cmp x9, #2
    bne NoWinoSteps
    mov x21, #4
    mul x15, x7, x8
    mul x15, x15, x21 // kernel_size * col *sizeof(float)
    mov x21, #32
    mul x16, x8, x21 // kernel_size * 8 * sizeof(float)
NoWinoSteps:
    mov x21, #4
    mul x8, x8, x21

LoopRowStart:
    cmp x9, #3
    bne LoopRow4
    mov x20, x2
LoopRow4:
    mov x14, x1 // reload rhs ptr
    mov x13, x7 // reload rhs col
    mov x12, x3 // reload bias

    LoopCol4:
        cbz x9, NoReloadDst4
        cmp x9, #3
        beq C4ReloadDst4
        mov x11, x2
        b NoReloadDst4
    C4ReloadDst4:
        mov x11, x20
    NoReloadDst4:
        mov x10, x0 // reload lhs ptr
        mov x19, x5 // reload depth

        cmp x13, #4
        ble LoopDepthStartHalf4

    LoopDepthStart4:
        ld1 {v0.4s}, [x10], #16
        ld1 {v3.4s, v4.4s}, [x14], #32
        fmul v8.4s, v3.4s, v0.s[0]
        fmul v10.4s, v3.4s, v0.s[1]
        fmul v12.4s, v3.4s, v0.s[2]
        fmul v14.4s, v3.4s, v0.s[3]
        fmul v9.4s, v4.4s, v0.s[0]
        fmul v11.4s, v4.4s, v0.s[1]
        fmul v13.4s, v4.4s, v0.s[2]
        fmul v15.4s, v4.4s, v0.s[3]

        subs x19, x19, #1
        beq Bias4

        LoopDepth4:
            ld1 {v0.4s}, [x10], #16
            ld1 {v3.4s, v4.4s}, [x14], #32
            fmla v8.4s, v3.4s, v0.s[0]
            fmla v10.4s, v3.4s, v0.s[1]
            fmla v12.4s, v3.4s, v0.s[2]
            fmla v14.4s, v3.4s, v0.s[3]
            fmla v9.4s, v4.4s, v0.s[0]
            fmla v11.4s, v4.4s, v0.s[1]
            fmla v13.4s, v4.4s, v0.s[2]
            fmla v15.4s, v4.4s, v0.s[3]

            subs x19, x19, #1
            bgt LoopDepth4

        Bias4:
            cbz x3, Activation4
            ld1 {v0.4s}, [x12], #16
            ld1 {v1.4s}, [x12], #16
            fadd v8.4s, v8.4s, v0.4s
            fadd v9.4s, v9.4s, v1.4s
            fadd v10.4s, v10.4s, v0.4s
            fadd v11.4s, v11.4s, v1.4s
            fadd v12.4s, v12.4s, v0.4s
            fadd v13.4s, v13.4s, v1.4s
            fadd v14.4s, v14.4s, v0.4s
            fadd v15.4s, v15.4s, v1.4s

        Activation4:
            cmp x4, #3
            beq Relu64
            cmp x4, #1
            beq Relu4
            b Write

        Relu64:
            mov w19, #6
            dup v2.4s, w19
            scvtf v2.4s, v2.4s
            fmin v8.4s, v8.4s, v2.4s
            fmin v9.4s, v9.4s, v2.4s
            fmin v10.4s, v10.4s, v2.4s
            fmin v11.4s, v11.4s, v2.4s
            fmin v12.4s, v12.4s, v2.4s
            fmin v13.4s, v13.4s, v2.4s
            fmin v14.4s, v14.4s, v2.4s
            fmin v15.4s, v15.4s, v2.4s

        Relu4:
            dup v3.4s, wzr
            fmax v8.4s, v8.4s, v3.4s
            fmax v9.4s, v9.4s, v3.4s
            fmax v10.4s, v10.4s, v3.4s
            fmax v11.4s, v11.4s, v3.4s
            fmax v12.4s, v12.4s, v3.4s
            fmax v13.4s, v13.4s, v3.4s
            fmax v14.4s, v14.4s, v3.4s
            fmax v15.4s, v15.4s, v3.4s
            b Write

    LoopDepthStartHalf4:
        ld1 {v0.4s}, [x10], #16
        ld1 {v3.4s}, [x14], #16
        ld1 {v4.4s}, [x14], #16
        fmul v8.4s, v3.4s, v0.s[0]
        fmul v10.4s, v3.4s, v0.s[1]
        fmul v12.4s, v3.4s, v0.s[2]
        fmul v14.4s, v3.4s, v0.s[3]

        subs x19, x19, #1
        beq BiasHalf4

        LoopDepthHalf4:
            ld1 {v0.4s}, [x10], #16
            ld1 {v3.4s}, [x14], #16
            ld1 {v4.4s}, [x14], #16
            fmla v8.4s, v3.4s, v0.s[0]
            fmla v10.4s, v3.4s, v0.s[1]
            fmla v12.4s, v3.4s, v0.s[2]
            fmla v14.4s, v3.4s, v0.s[3]

            subs x19, x19, #1
            bgt LoopDepthHalf4

        BiasHalf4:
            cbz x3, ActivationHalf4
            ld1 {v0.4s}, [x12], #16
            ld1 {v1.4s}, [x12], #16
            fadd v8.4s, v8.4s, v0.4s
            fadd v10.4s, v10.4s, v0.4s
            fadd v12.4s, v12.4s, v0.4s
            fadd v14.4s, v14.4s, v0.4s

        ActivationHalf4:
            cmp x4, #3
            beq Relu6Half4
            cmp x4, #1
            beq ReluHalf4
            b Write

        Relu6Half4:
            mov w19, #6
            dup v2.4s, w19
            scvtf v2.4s, v2.4s
            fmin v8.4s, v8.4s, v2.4s
            fmin v10.4s, v10.4s, v2.4s
            fmin v12.4s, v12.4s, v2.4s
            fmin v14.4s, v14.4s, v2.4s

        ReluHalf4:
            dup v3.4s, wzr
            fmax v8.4s, v8.4s, v3.4s
            fmax v10.4s, v10.4s, v3.4s
            fmax v12.4s, v12.4s, v3.4s
            fmax v14.4s, v14.4s, v3.4s

        Write:
            cmp x9, #2
            beq WriteWino
            cmp x9, #3
            beq WriteC4
            cbz x9, WriteC8
            cmp x13, #1
            beq Write1
            cmp x13, #2
            beq Write2
            cmp x13, #3
            beq Write3
            cmp x13, #4
            beq Write4
            cmp x13, #5
            beq Write5
            cmp x13, #6
            beq Write6
            cmp x13, #7
            beq Write7
            b Write8

        Write1:
            add x2, x2, #4
            str s8, [x11]
            cmp x6, #1
            beq WriteEnd
            add x11, x11, x8
            str s10, [x11]
            cmp x6, #2
            beq WriteEnd
            add x11, x11, x8
            str s12, [x11]
            cmp x6, #3
            beq WriteEnd
            add x11, x11, x8
            str s14, [x11]
            add x11, x11, x8
            add x11, x11, #4
            b WriteEnd
        Write2:
            add x2, x2, #8
            st1 {v8.2s}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.2s}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.2s}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.2s}, [x11], x8
            add x11, x11, #8
            b WriteEnd
        Write3:
            add x2, x2, #12
            add x19, x11, #8
            st1 {v8.2s}, [x11], x8
            st1 {v8.s}[2], [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.2s}, [x11], x8
            st1 {v10.s}[2], [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.2s}, [x11], x8
            st1 {v12.s}[2], [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.2s}, [x11], x8
            st1 {v14.s}[2], [x19]
            add x11, x11, #12
            b WriteEnd
        Write4:
            add x2, x2, #16
            st1 {v8.4s}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], x8
            add x11, x11, #16
            b WriteEnd
        Write5:
            add x2, x2, #20
            add x19, x11, #16
            st1 {v8.4s}, [x11], x8
            str s9, [x19]
            cmp x6, #1
            beq WriteEnd
            add x19, x19, x8
            st1 {v10.4s}, [x11], x8
            str s11, [x19]
            cmp x6, #2
            beq WriteEnd
            add x19, x19, x8
            st1 {v12.4s}, [x11], x8
            str s13, [x19]
            cmp x6, #3
            beq WriteEnd
            add x19, x19, x8
            st1 {v14.4s}, [x11], x8
            str s15, [x19]
            add x11, x11, #20
            b WriteEnd
        Write6:
            add x2, x2, #24
            add x19, x11, #16
            st1 {v8.4s}, [x11], x8
            st1 {v9.2s}, [x19], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], x8
            st1 {v11.2s}, [x19], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], x8
            st1 {v13.2s}, [x19], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], x8
            st1 {v15.2s}, [x19], x8
            add x11, x11, #24
            b WriteEnd
        Write7:
            add x2, x2, #28
            add x19, x11, #16
            add x20, x11, #24
            st1 {v8.4s}, [x11], x8
            st1 {v9.2s}, [x19], x8
            st1 {v9.s}[2], [x20], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], x8
            st1 {v11.2s}, [x19], x8
            st1 {v11.s}[2], [x20], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], x8
            st1 {v13.2s}, [x19], x8
            st1 {v13.s}[2], [x20], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], x8
            st1 {v15.2s}, [x19], x8
            st1 {v15.s}[2], [x20], x8
            add x11, x11, #28
            b WriteEnd
        WriteC8:
            mov x19, x11
            st1 {v8.4s, v9.4s, v10.4s, v11.4s}, [x19], #64
            st1 {v12.4s, v13.4s, v14.4s, v15.4s}, [x19], #64

            add x11, x11, x16
            b WriteEnd
        WriteWino:
            add x2, x11, x16
            st1 {v8.4s, v9.4s}, [x11], x15
            st1 {v10.4s, v11.4s}, [x11], x15
            st1 {v12.4s, v13.4s}, [x11], x15
            st1 {v14.4s, v15.4s}, [x11], x15

            b WriteEnd
        Write8:
            add x2, x2, #32
            st1 {v8.4s, v9.4s}, [x11], x8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s, v11.4s}, [x11], x8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s, v13.4s}, [x11], x8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s, v15.4s}, [x11], x8
            add x11, x11, #32
            b WriteEnd
        WriteC4:
            cmp x13, #1
            beq C4Write1
            cmp x13, #2
            beq C4Write2
            cmp x13, #3
            beq C4Write3
            cmp x13, #4
            beq C4Write4
            cmp x13, #5
            beq C4Write5
            cmp x13, #6
            beq C4Write6
            cmp x13, #7
            beq C4Write7
            b C4Write8
        C4Write1:
            str s8, [x11], #4
            cmp x6, #1
            beq WriteEnd
            str s10, [x11], #4
            cmp x6, #2
            beq WriteEnd
            str s12, [x11], #4
            cmp x6, #3
            beq WriteEnd
            str s14, [x11], #4
            b WriteEnd
        C4Write2:
            st1 {v8.2s}, [x11], #8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.2s}, [x11], #8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.2s}, [x11], #8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.2s}, [x11], #8
            b WriteEnd
        C4Write3:
            add x19, x11, #8
            st1 {v8.2s}, [x11]
            add x11, x11, #12
            st1 {v8.s}[2], [x19]
            add x19, x19, #12
            cmp x6, #1
            beq WriteEnd
            st1 {v10.2s}, [x11]
            add x11, x11, #12
            st1 {v10.s}[2], [x19]
            add x19, x19, #12
            cmp x6, #2
            beq WriteEnd
            st1 {v12.2s}, [x11]
            add x11, x11, #12
            st1 {v12.s}[2], [x19]
            add x19, x19, #12
            cmp x6, #3
            beq WriteEnd
            st1 {v14.2s}, [x11]
            st1 {v14.s}[2], [x19]
            b WriteEnd
        C4Write4:
            st1 {v8.4s}, [x11], #16
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], #16
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], #16
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], #16
            b WriteEnd
        C4Write5:
            add x19, x11, x8
            st1 {v8.4s}, [x11], #16
            str s9, [x19], #4
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], #16
            str s11, [x19], #4
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], #16
            str s13, [x19], #4
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], #16
            str s15, [x19], #4
            b WriteEnd
        C4Write6:
            add x19, x11, x8
            st1 {v8.4s}, [x11], #16
            st1 {v9.2s}, [x19], #8
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], #16
            st1 {v11.2s}, [x19], #8
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], #16
            st1 {v13.2s}, [x19], #8
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], #16
            st1 {v15.2s}, [x19], #8
            b WriteEnd
        C4Write7:
            add x19, x11, x8
            add x16, x19, #8
            mov x15, #12
            st1 {v8.4s}, [x11], #16
            st1 {v9.2s}, [x19], x15
            st1 {v9.s}[2], [x16], x15
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], #16
            st1 {v11.2s}, [x19], x15
            st1 {v11.s}[2], [x16], x15
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], #16
            st1 {v13.2s}, [x19], x15
            st1 {v13.s}[2], [x16], x15
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], #16
            st1 {v15.2s}, [x19], x15
            st1 {v15.s}[2], [x16], x15
            b WriteEnd
        C4Write8:
            add x19, x11, x8
            add x20, x19, x8
            st1 {v8.4s}, [x11], #16
            st1 {v9.4s}, [x19], #16
            cmp x6, #1
            beq WriteEnd
            st1 {v10.4s}, [x11], #16
            st1 {v11.4s}, [x19], #16
            cmp x6, #2
            beq WriteEnd
            st1 {v12.4s}, [x11], #16
            st1 {v13.4s}, [x19], #16
            cmp x6, #3
            beq WriteEnd
            st1 {v14.4s}, [x11], #16
            st1 {v15.4s}, [x19], #16
        WriteEnd:
            subs x13, x13, #8 // rhs col - 8
            bgt LoopCol4


LoopColEnd:
        add x0, x0, x17
        cbz x9, C8DstStep
        cmp x9, #3
        beq C4DstStep
        mov x21, #4
        mul x21, x21, x7
        sub x11, x11, x21
        mov x2, x11
        b NoDstStep
    C4DstStep:
        add x2, x2, x18
        b NoDstStep
    C8DstStep:
        add x2, x2, #384
        mov x11, x2
    NoDstStep:
        subs x6, x6, #12
        bgt LoopRow4

  ld1 {v8.4s, v9.4s, v10.4s, v11.4s}, [sp], #64
  ld1 {v12.4s, v13.4s, v14.4s, v15.4s}, [sp], #64
  ldp x19, x20, [sp], #16
  ldp x21, x22, [sp], #16
  ret
#endif
