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
#include "infer/ops_func_impl/kl_div_grad.h"
#include <set>
#include <memory>
#include <string>
#include <vector>
#include <algorithm>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {
ShapeArray KLDivGradFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex1];
  const auto &input_shape = input->GetShape();
  return {input_shape};
}

std::vector<TypeId> KLDivGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  const std::vector<std::string> arg_names{"grad_output", "input", "target"};
  const auto &prim_name = primitive->name();
  std::vector<TypeId> arg_types;
  (void)std::transform(input_infos.begin(), input_infos.begin() + kInputIndex3, std::back_inserter(arg_types),
                       [](const auto &info) { return info->GetType(); });
  for (size_t i = 0; i < arg_names.size(); ++i) {
    (void)CheckAndConvertUtils::CheckTypeIdValid(arg_names[i], arg_types[i], valid_types, prim_name);
  }
  return {arg_types[kInputIndex1]};
}

}  // namespace ops
}  // namespace mindspore
