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
REG_CUST_OP(SparseApplyCenteredRMSProp)
  .INPUT(var, TensorType::RealNumberType())
  .INPUT(mg, TensorType::RealNumberType())
  .INPUT(ms, TensorType::RealNumberType())
  .INPUT(mom, TensorType::RealNumberType())
  .INPUT(lr, TensorType::RealNumberType())
  .INPUT(rho, TensorType::RealNumberType())
  .INPUT(momentum, TensorType::RealNumberType())
  .INPUT(epsilon, TensorType::RealNumberType())
  .INPUT(grad, TensorType::RealNumberType())
  .INPUT(indices, TensorType::RealIndexNumberType())
  .OUTPUT(var, TensorType::RealNumberType())
  .ATTR(use_locking, Bool, false)
  .CUST_OP_END_FACTORY_REG(SparseApplyCenteredRMSProp)
}  // namespace ge

#endif  // CUSTOMIZE_OP_PROTO_INC_CHOLESKY_SOLVE_OP_H