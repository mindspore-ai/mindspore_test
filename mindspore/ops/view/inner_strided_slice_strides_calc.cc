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
#include "view/inner_strided_slice_strides_calc.h"
#include <vector>
#include <memory>
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore::ops {
namespace {
void InnerStridedSliceConvertNegToPos(std::vector<int64_t> *begin, std::vector<int64_t> *end,
                                      const std::vector<int64_t> &tensor_shape) {
  if (begin->size() != tensor_shape.size()) {
    MS_EXCEPTION(ValueError) << "Convert shape size is not equal";
  }
  if (end->size() != tensor_shape.size()) {
    MS_EXCEPTION(ValueError) << "Convert shape size is not equal";
  }
  for (size_t i = 0; i < tensor_shape.size(); ++i) {
    if ((*begin)[i] < 0) {
      (*begin)[i] += tensor_shape[i];
    }
    if ((*end)[i] < 0) {
      (*end)[i] += tensor_shape[i];
    }
    if ((*begin)[i] < 0) {
      (*begin)[i] = 0;
    } else if ((*begin)[i] >= tensor_shape[i]) {
      (*begin)[i] = tensor_shape[i];
    }
    if ((*end)[i] < (*begin)[i]) {
      (*end)[i] = (*begin)[i];
    } else if ((*end)[i] >= tensor_shape[i]) {
      (*end)[i] = tensor_shape[i];
    }
    if ((*begin)[i] == (*end)[i]) {
      (*begin)[i] = 0;
      (*end)[i] = 0;
    }
  }
}
void InnerStridedSliceVectorEmplace(std::vector<int64_t> *vec, size_t number, size_t dst_size) {
  if ((*vec).size() >= dst_size) {
    return;
  }
  auto num = dst_size - vec->size();
  for (size_t i = 0; i < num; ++i) {
    (void)vec->emplace_back(number);
  }
}

void InnerStridedSliceVectorEmplace(std::vector<int64_t> *vec, std::vector<int64_t> *number_vec, size_t dst_size) {
  if ((*vec).size() >= dst_size) {
    return;
  }

  if (number_vec->size() != dst_size) {
    MS_LOG(EXCEPTION) << "dst_size is not equal to number_vec.size(), dst_size:" << dst_size
                      << ",  number_vec.size():" << number_vec->size();
  }

  auto begin = vec->size();
  for (size_t i = begin; i < dst_size; ++i) {
    (void)vec->emplace_back(number_vec->at(i));
  }
}

bool CheckInputsNull(const std::vector<ValuePtr> &inputs) {
  bool nullptr_input_exist =
    std::any_of(inputs.cbegin(), inputs.cend(), [](const ValuePtr &v) { return v == nullptr; });
  return nullptr_input_exist;
}
}  // namespace

TensorStorageInfoPtrList InnerStridedSliceStridesCalc(const OldTensorInfoPtr old_tensor_info, size_t size,
                                                      std::vector<int64_t> *shape, std::vector<int64_t> *begin,
                                                      std::vector<int64_t> *end, std::vector<int64_t> *step) {
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto old_storage_offset = old_tensor_info->old_offset;

  InnerStridedSliceVectorEmplace(begin, size_t(0), size);
  InnerStridedSliceVectorEmplace(end, shape, size);
  InnerStridedSliceVectorEmplace(step, 1, size);
  InnerStridedSliceConvertNegToPos(begin, end, old_shape);

  for (size_t i = 0; i < begin->size(); ++i) {
    old_storage_offset += LongToSize(begin->at(i) * old_strides[i]);
  }
  ShapeVector new_shape;
  auto new_strides = old_strides;
  for (size_t i = 0; i < size; ++i) {
    auto dim = DynamicDimWrap(i, shape->size());
    auto real_end = end->at(dim) > old_shape[dim] ? old_shape[dim] : end->at(dim);
    auto len = real_end - begin->at(dim);
    if (len <= 0) {
      (void)new_shape.emplace_back(0);
    } else {
      auto shape_dim = (len + step->at(dim) - 1) / step->at(dim);
      (void)new_shape.emplace_back(shape_dim);
    }
    new_strides[dim] *= step->at(dim);
  }

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList InnerStridedSliceCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs)) {
    return {};
  }
  if (inputs[kInputIndex1]->isa<tensor::BaseTensor>() || inputs[kInputIndex2]->isa<tensor::BaseTensor>() ||
      inputs[kInputIndex3]->isa<tensor::BaseTensor>()) {
    return {};
  }
  auto begin = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  auto end = GetValue<std::vector<int64_t>>(inputs[kInputIndex2]);
  auto step = GetValue<std::vector<int64_t>>(inputs[kInputIndex3]);
  if (IsDynamic(step) || begin.size() != end.size() || begin.size() != step.size() || HasZero(step)) {
    return {};
  }
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto size = input_tensor->shape().size();
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto shape = input_tensor->shape();
  return InnerStridedSliceStridesCalc(old_tensor_info, size, &shape, &begin, &end, &step);
}

REG_VIEW_STRIDES_CALC_FUN(InnerStridedSlice, InnerStridedSliceCalc);
}  // namespace mindspore::ops
