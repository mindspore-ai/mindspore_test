/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "mindspore/ccsrc/frontend/operator/meta_dsl/func_op/softmax_ext.h"
#include "mindapi/base/type_id.h"
#include "ir/dtype/type.h"
#include "utils/shape_utils.h"
#include "utils/core_op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/ops_utils/op_constants.h"

namespace mindspore::prim {
void CheckSoftmaxExtInputs(const PrimitivePtr &primitive, const AbstractBasePtrList &input_args) {
  const std::set<TypePtr> valid_types{kBFloat16, kFloat16, kFloat32, kFloat64};
  const auto input_type = input_args[kIndex0]->GetType();

  if (input_args.size() < kIndex1 || input_args[kIndex1]->GetValue() == nullptr) {
    MS_LOG(EXCEPTION) << "In SoftmaxExt, dim should not be nullptr.";
  }
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input", input_type, valid_types, primitive->name());
}

BeginFunction(SoftmaxExt, input, dim, dtype) {
  auto axis = Tuple(dim);
  auto true_branch = [&]() {
    auto converted_tensor = Call(Prim(Cast), input, dtype);
    Return(Call(Prim(Softmax), converted_tensor, axis));
  };
  auto false_branch = [&]() { Return(Call(Prim(Softmax), input, axis)); };

  Return(If(IsNotNone(dtype), true_branch, false_branch, (input, axis, dtype)));
}
EndFunction(SoftmaxExt)
}  // namespace mindspore::prim
