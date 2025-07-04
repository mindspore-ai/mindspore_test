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

#include "infer/ops_func_impl/eltwise_grad_op.h"
#include <vector>
#include <memory>
#include "infer/ops_func_impl/common_infer_fns.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace ops {
BaseShapePtr EltwiseGradOpFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return BinaryOpShapesEqualInfer(primitive, input_args);
}

TypePtr EltwiseGradOpFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
