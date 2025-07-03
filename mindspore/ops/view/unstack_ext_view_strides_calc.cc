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
#include "view/unstack_ext_view_strides_calc.h"
#include <vector>
#include <memory>
#include <set>
#include <string>
#include "view/unstack_strides_calc.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
TensorStorageInfoPtrList UnstackExtViewBasicTypeCalc(const tensor::TensorPtr &x_tensor, const int64_t &dim) {
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto type = x_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", type, common_valid_types_with_complex_and_bool, "UnstackExt");
  auto old_tensor_info = GetOldTensorInfo(x_tensor);
  return UnstackStridesCalc(old_tensor_info, dim);
}

TensorStorageInfoPtrList UnstackExtViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>()) {
    return {};
  }
  auto tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  auto type = tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", type, common_valid_types_with_complex_and_bool, "UnstackExtView");
  auto dim_value_ptr = inputs[kInputIndex1];
  MS_EXCEPTION_IF_NULL(dim_value_ptr);
  auto dim = GetValue<int64_t>(dim_value_ptr);
  return UnstackExtViewBasicTypeCalc(inputs[kInputIndex0]->cast<tensor::TensorPtr>(), dim);
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(UnstackExtView, UnstackExtViewCalc);

}  // namespace mindspore::ops
