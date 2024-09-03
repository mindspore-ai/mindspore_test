/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/lstsq_v2_grad.h"
#include <memory>
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LstsqV2GradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto input_a_shape = input_args[kIndex1]->GetShape();
  auto input_b_shape = input_args[kIndex2]->GetShape();

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{input_a_shape->Clone(), input_b_shape->Clone()});
}

TypePtr LstsqV2GradFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  auto input_a_type = input_args[kIndex0]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{input_a_type->Clone(), input_a_type->Clone()});
}
}  // namespace ops
}  // namespace mindspore
