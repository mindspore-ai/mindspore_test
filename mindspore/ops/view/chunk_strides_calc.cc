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
#include <algorithm>
#include <memory>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/chunk_strides_calc.h"

namespace mindspore::ops {
void ChunkInputsCheck(const PrimitivePtr &prim, const int64_t &output_num, const int64_t &axis) {
  auto prim_name = prim->name();
  if (output_num <= 0) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', output_num must be positive, but got " << output_num << ".";
  }
}

TensorStorageInfoPtrList ChunkCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::BaseTensor>()) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", first inputs must be a Tensor, but got: " << inputs[kInputIndex0]->ToString();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto chunks = GetValue<int64_t>(inputs[kInputIndex1]);
  MS_CHECK_VALUE(chunks > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("chunks", chunks, kGreaterEqual, 1, prim));
  auto dim = GetValue<int64_t>(inputs[kInputIndex2]);
  auto tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(tensor_info);
  auto old_shape = tensor_info->old_shape;
  auto old_strides = tensor_info->old_strides;

  auto rank = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(rank > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("rank", rank, kGreaterEqual, 1, prim));

  const auto ndim = old_shape.size();
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  int64_t dim_size = old_shape[wrap_dim];
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  if (dim_size == 0) {
    if (split_size == 0) {
      auto storage_info = input_tensor->storage_info();
      if (storage_info == nullptr) {
        storage_info = std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_shape, old_strides,
                                                           IsContiguous(old_shape, old_strides));
      }
      return {storage_info};
    }
    MS_EXCEPTION(ValueError) << "For '" << prim->name() << "', output_num must be positive, but got 0.";
  }

  // Calculate the number of sub tensors after segmentation
  auto num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  auto last_split_size = split_size - (split_size * num_splits - dim_size);
  // Create a storage information list
  std::vector<TensorStorageInfoPtr> storage_info_list;

  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = old_shape;

    // Calculate the size of a sub tensor in a specified dimension
    slice_shape[wrap_dim] = (idx == num_splits - 1) ? last_split_size : split_size;
    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = tensor_info->old_offset + LongToSize(idx * split_size * old_strides[wrap_dim]);
    auto new_storage_info =
      std::make_shared<TensorStorageInfo>(slice_shape, old_strides, new_storage_offset, tensor_info->ori_shape,
                                          tensor_info->ori_strides, IsContiguous(slice_shape, old_strides));
    storage_info_list.emplace_back(new_storage_info);
  }
  return storage_info_list;
}
REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Chunk, ChunkCalc);
}  // namespace mindspore::ops
