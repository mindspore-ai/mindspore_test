/**
 * Copyright (c) 2022-2022 Huawei Technologies Co., Ltd.  All rights reserved.
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

#ifndef CUSTOMIZE_OP_PROTO_INC_SPARSE_MATRIX_TRANSPOSE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_SPARSE_MATRIX_TRANSPOSE_OP_H

#include "op_proto_macro.h"

namespace ge {
REG_CUST_OP(SparseMatrixTranspose)
  .INPUT(x_dense_shape, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x_batch_pointers, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x_row_pointers, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x_col_indices, TensorType({DT_INT32, DT_INT64}))
  .INPUT(x_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64,
                               DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .OUTPUT(y_dense_shape, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_batch_pointers, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_row_pointers, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_col_indices, TensorType({DT_INT32, DT_INT64}))
  .OUTPUT(y_values, TensorType({DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32, DT_UINT64,
                                DT_FLOAT, DT_DOUBLE, DT_COMPLEX64, DT_COMPLEX128}))
  .ATTR(conjugate, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseMatrixTranspose)
}  // namespace ge
#endif