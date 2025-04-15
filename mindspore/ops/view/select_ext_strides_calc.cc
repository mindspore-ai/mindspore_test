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
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/select_ext_strides_calc.h"

namespace {
constexpr size_t kSelectExtInputsNum = 3;
}

namespace mindspore::ops {

TensorStorageInfoPtrList SelectExtStridesCalc(const OldTensorInfoPtr old_tensor_info, const int64_t ori_dim,
                                              const int64_t ori_index) {
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  int dim_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(dim_size > 0, "For Primitive [SelectExt] rank must >= 1");

  auto dim = DynamicDimWrap(ori_dim, dim_size);
  auto dim_value = old_shape[dim];

  MS_CHECK_VALUE(ori_index >= -dim_value && ori_index < dim_value,
                 "For Primitive [SelectExt] start exceed range. start: " + std::to_string(ori_index) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + ").");
  auto index = ori_index < 0 ? ori_index + dim_value : ori_index;

  auto new_shape = old_shape;
  auto new_strides = old_strides;
  size_t new_storage_offset = old_storage_offset;
  new_storage_offset += LongToSize(index * old_strides[dim]);
  new_shape.erase(new_shape.begin() + dim);
  new_strides.erase(new_strides.begin() + dim);

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}
TensorStorageInfoPtrList SelectExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSelectExtInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "inputs num is invalid, num:" << inputs.size();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type = input_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, common_valid_types_with_complex_and_bool,
                                             prim->name());
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto dim = GetValue<int64_t>(inputs[kInputIndex1]);
  auto index = GetValue<int64_t>(inputs[kInputIndex2]);
  return SelectExtStridesCalc(old_tensor_info, dim, index);
}

REG_VIEW_STRIDES_CALC_FUN(SelectExt, SelectExtCalc);
}  // namespace mindspore::ops
