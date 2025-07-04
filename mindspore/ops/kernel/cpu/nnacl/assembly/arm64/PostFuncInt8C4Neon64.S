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

//void PostFuncInt8C4Neon64(const int32_t *in, const int32_t *bias, int8_t *out, size_t oc4div, size_t oc4res,
//                          size_t plane, size_t stride, int32_t multiplier, int32_t left_shift, int32_t right_shift,
//                          int32_t zp, int32_t mini, int32_t maxi);
// x0 in
// x1 bias
// x2 out
// x3 oc4div
// x4 oc4res
// x5 plane
// x6 stride
// x7 multiplier
// x8 left_shift
// x9 right_shift
// x10 zp
// x11 mini
// x12 maxi

// v0 ~ v15 value
// x24  x25  write loop tmp buf


// v16  bias data

// v26  multiplier
// v27  left_shift
// v28  right_shift
// v29  zp
// v30  min
// v31  max

// w15  oc4 loop control
// w16  hw  loop control

asm_function PostFuncInt8C4Neon64
  sub sp, sp, #16
  stp x24, x25, [sp]

  ldr w8, [sp, #16]
  ldr w9, [sp, #24]
  ldr w10, [sp, #32]
  ldr w11, [sp, #40]
  ldr w12, [sp, #48]
  ldr w13, [sp, #56]

  dup v26.4s, w7
  dup v27.4s, w8
  dup v28.4s, w9
  dup v29.4s, w10
  dup v30.4s, w11
  dup v31.4s, w12

  mov x15, #0

Loop_C4:
  cmp x15, x3
  beq Loop_C1
  mov x25,  #4
  mul x24, x15, x25
  add x25, x2, x24
  add w15, w15, #4
  mov w16, w5
  ld1 {v16.4s}, [x1], #16

Loop_4x4:
  cmp x16, #4
  blt Loop_1x4
  sub x16, x16, #4
  ld1 {v0.4s, v1.4s, v2.4s, v3.4s}, [x0], #64

  add v0.4s, v0.4s, v16.4s
  add v1.4s, v1.4s, v16.4s
  add v2.4s, v2.4s, v16.4s
  add v3.4s, v3.4s, v16.4s
  sqshl v0.4s, v0.4s, v27.4s
  sqshl v1.4s, v1.4s, v27.4s
  sqshl v2.4s, v2.4s, v27.4s
  sqshl v3.4s, v3.4s, v27.4s
  sqrdmulh v0.4s, v0.4s, v26.4s
  sqrdmulh v1.4s, v1.4s, v26.4s
  sqrdmulh v2.4s, v2.4s, v26.4s
  sqrdmulh v3.4s, v3.4s, v26.4s
  and v4.16b, v28.16b, v0.16b
  and v5.16b, v28.16b, v1.16b
  and v6.16b, v28.16b, v2.16b
  and v7.16b, v28.16b, v3.16b
  sshr v4.4s, v4.4s, #31
  sshr v5.4s, v5.4s, #31
  sshr v6.4s, v6.4s, #31
  sshr v7.4s, v7.4s, #31
  sqadd v0.4s, v0.4s, v4.4s
  sqadd v1.4s, v1.4s, v5.4s
  sqadd v2.4s, v2.4s, v6.4s
  sqadd v3.4s, v3.4s, v7.4s
  srshl v0.4s, v0.4s, v28.4s
  srshl v1.4s, v1.4s, v28.4s
  srshl v2.4s, v2.4s, v28.4s
  srshl v3.4s, v3.4s, v28.4s
  add v0.4s, v0.4s, v29.4s
  add v1.4s, v1.4s, v29.4s
  add v2.4s, v2.4s, v29.4s
  add v3.4s, v3.4s, v29.4s
  smax v0.4s, v0.4s, v30.4s
  smax v1.4s, v1.4s, v30.4s
  smax v2.4s, v2.4s, v30.4s
  smax v3.4s, v3.4s, v30.4s
  smin v0.4s, v0.4s, v31.4s
  smin v1.4s, v1.4s, v31.4s
  smin v2.4s, v2.4s, v31.4s
  smin v3.4s, v3.4s, v31.4s
  sqxtn v4.4h, v0.4s
  sqxtn v5.4h, v1.4s
  sqxtn v6.4h, v2.4s
  sqxtn v7.4h, v3.4s
  sqxtn v0.8b, v4.8h
  sqxtn v1.8b, v5.8h
  sqxtn v2.8b, v6.8h
  sqxtn v3.8b, v7.8h

  st1 {v0.s}[0], [x25], x6
  st1 {v1.s}[0], [x25], x6
  st1 {v2.s}[0], [x25], x6
  st1 {v3.s}[0], [x25], x6
  b Loop_4x4


Loop_1x4:
  cmp x16, #0
  beq Loop_C4
  sub x16, x16, #1
  ld1 {v0.4s}, [x0], #16

  add v0.4s, v0.4s, v16.4s
  sqshl v0.4s, v0.4s, v27.4s
  sqrdmulh v0.4s, v0.4s, v26.4s
  and v2.16b, v28.16b, v0.16b
  sshr v2.4s, v2.4s, #31
  sqadd v0.4s, v0.4s, v2.4s
  srshl v0.4s, v0.4s, v28.4s
  add v0.4s, v0.4s, v29.4s
  smax v0.4s, v0.4s, v30.4s
  smin v0.4s, v0.4s, v31.4s
  sqxtn v1.4h, v0.4s
  sqxtn v0.8b, v1.8h

  st1 {v0.s}[0], [x25], x6
  b Loop_1x4

Loop_C1:
  cmp x4, #0
  beq End
  mov x16, x5
  ld1 {v16.4s}, [x1], #16
  mov x25,  #4
  mul x24, x15, x25
  add x25, x2, x24
  add x24, x25, #2

  cmp x4, #1
  beq Loop_C1_1
  cmp x4, #2
  beq Loop_C1_2
  cmp x4, #3
  beq Loop_C1_3

Loop_C1_1:
  cmp x16, #0
  beq End
  sub x16, x16, #1
  ld1 {v0.4s}, [x0], #16

  add v0.4s, v0.4s, v16.4s
  sqshl v0.4s, v0.4s, v27.4s
  sqrdmulh v0.4s, v0.4s, v26.4s
  and v2.16b, v28.16b, v0.16b
  sshr v2.4s, v2.4s, #31
  sqadd v0.4s, v0.4s, v2.4s
  srshl v0.4s, v0.4s, v28.4s
  add v0.4s, v0.4s, v29.4s
  smax v0.4s, v0.4s, v30.4s
  smin v0.4s, v0.4s, v31.4s
  sqxtn v1.4h, v0.4s
  sqxtn v0.8b, v1.8h

  st1 {v0.b}[0], [x25], x6
  b Loop_C1_1


Loop_C1_2:
  cmp x16, #0
  beq End
  sub x16, x16, #1
  ld1 {v0.4s}, [x0], #16

  add v0.4s, v0.4s, v16.4s
  sqshl v0.4s, v0.4s, v27.4s
  sqrdmulh v0.4s, v0.4s, v26.4s
  and v2.16b, v28.16b, v0.16b
  sshr v2.4s, v2.4s, #31
  sqadd v0.4s, v0.4s, v2.4s
  srshl v0.4s, v0.4s, v28.4s
  add v0.4s, v0.4s, v29.4s
  smax v0.4s, v0.4s, v30.4s
  smin v0.4s, v0.4s, v31.4s
  sqxtn v1.4h, v0.4s
  sqxtn v0.8b, v1.8h

  st1 {v0.h}[0], [x25], x6
  b Loop_C1_2


Loop_C1_3:
  cmp x16, #0
  beq End
  sub x16, x16, #1
  ld1 {v0.4s}, [x0], #16

  add v0.4s, v0.4s, v16.4s
  sqshl v0.4s, v0.4s, v27.4s
  sqrdmulh v0.4s, v0.4s, v26.4s
  and v2.16b, v28.16b, v0.16b
  sshr v2.4s, v2.4s, #31
  sqadd v0.4s, v0.4s, v2.4s
  srshl v0.4s, v0.4s, v28.4s
  add v0.4s, v0.4s, v29.4s
  smax v0.4s, v0.4s, v30.4s
  smin v0.4s, v0.4s, v31.4s
  sqxtn v1.4h, v0.4s
  sqxtn v0.8b, v1.8h

  st1 {v0.h}[0], [x25], x6
  st1 {v0.b}[2], [x24], x6
  b Loop_C1_3


End:
  ldp x24, x25, [sp], #16
  ret
#endif
