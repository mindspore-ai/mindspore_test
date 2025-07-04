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

#ifndef CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H
#define CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H

#include "op_proto_macro.h"

namespace ge {
REG_CUST_OP(InstanceNormV2Grad)
  .INPUT(dy, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .INPUT(gamma, TensorType({DT_FLOAT}))
  .INPUT(mean, TensorType({DT_FLOAT}))
  .INPUT(variance, TensorType({DT_FLOAT}))
  .INPUT(save_mean, TensorType({DT_FLOAT}))
  .INPUT(save_variance, TensorType({DT_FLOAT}))

  .OUTPUT(pd_x, TensorType({DT_FLOAT16, DT_FLOAT}))
  .OUTPUT(pd_gamma, TensorType({DT_FLOAT}))
  .OUTPUT(pd_beta, TensorType({DT_FLOAT}))

  .ATTR(is_training, Bool, true)
  .ATTR(epsilon, Float, 0.00001)
  .CUST_OP_END_FACTORY_REG(InstanceNormV2Grad)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H