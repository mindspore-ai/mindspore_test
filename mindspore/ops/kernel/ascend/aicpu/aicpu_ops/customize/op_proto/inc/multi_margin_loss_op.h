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

#ifndef CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_OP_H
#define CUSTOMIZE_OP_PROTO_INC_MULTI_MARGIN_LOSS_OP_H

#include "op_proto_macro.h"

namespace ge {
REG_CUST_OP(MultiMarginLoss)
  .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .INPUT(target, TensorType({DT_INT64}))
  .OPTIONAL_INPUT(weight, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .ATTR(p, Int, 1)
  .ATTR(margin, Float, 1.0)
  .ATTR(reduction, String, "mean")
  .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE}))
  .CUST_OP_END_FACTORY_REG(MultiMarginLoss)
}  // namespace ge
#endif