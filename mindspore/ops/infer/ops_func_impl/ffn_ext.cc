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
#include <memory>
#include <set>
#include "infer/ops_func_impl/ffn_ext.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFNExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_shape = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  return std::make_shared<abstract::TensorShape>(x_shape);
}

TypePtr FFNExtFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16, kInt8};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, primitive->name());
  return x_type;
}
}  // namespace ops
}  // namespace mindspore
