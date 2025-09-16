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
#include <utility>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/chunk_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList ChunkStridesCalc(const std::vector<int64_t> &old_shape,
                                          const std::vector<int64_t> &old_strides,
                                          const TensorStorageInfoPtr &storage_info, const int64_t &chunks,
                                          const int64_t &dim) {
  const auto ndim = old_shape.size();
  if (ndim == 0) {
    MS_LOG(EXCEPTION) << "For 'Chunk', rank should > 0, but get " << ndim;
  }
  auto [ori_shape, ori_strides, old_offset] = GetOriShapeStridesAndOffset(old_shape, old_strides, storage_info);
  const auto wrap_dim = DynamicDimWrap(dim, ndim);
  int64_t dim_size = old_shape[wrap_dim];
  int64_t split_size = (dim_size + chunks - 1) / chunks;

  if (MS_UNLIKELY(dim_size == 0)) {
    if (split_size == 0) {
      TensorStorageInfoPtr new_storage_info{storage_info};
      if (storage_info == nullptr) {
        new_storage_info = std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_shape, old_strides,
                                                               IsContiguous(old_shape, old_strides));
      }
      std::vector<TensorStorageInfoPtr> storage_info_list(chunks, new_storage_info);
      return storage_info_list;
    }
    MS_EXCEPTION(ValueError) << "For 'Chunk', output_num must be positive, but got 0.";
  }

  // Calculate the number of sub tensors after segmentation
  auto num_splits = std::max<int64_t>((dim_size + split_size - 1) / split_size, 1);
  auto last_split_size = split_size - (split_size * num_splits - dim_size);
  // Create a storage information list
  std::vector<TensorStorageInfoPtr> storage_info_list{};

  for (int64_t idx = 0; idx < num_splits; ++idx) {
    // Calculate the shape and length of sub tensors
    std::vector<int64_t> slice_shape = old_shape;

    // Calculate the size of a sub tensor in a specified dimension
    slice_shape[wrap_dim] = (idx == num_splits - 1) ? last_split_size : split_size;
    // Calculate the storage offset of sub tensors
    size_t new_storage_offset = old_offset + LongToSize(idx * split_size * old_strides[wrap_dim]);
    bool is_contiguous = IsContiguous(slice_shape, old_strides);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(std::move(slice_shape), old_strides, new_storage_offset,
                                                                ori_shape, ori_strides, is_contiguous);
    (void)storage_info_list.emplace_back(new_storage_info);
  }

  return storage_info_list;
}

TensorStorageInfoPtrList ChunkBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &chunks,
                                            const int64_t &dim) {
  if (MS_UNLIKELY(chunks < 1)) {
    MS_EXCEPTION(ValueError) << "For 'Chunk', chunks should be greater equal than 1, but got " << chunks;
  }
  MS_EXCEPTION_IF_NULL(input_tensor);
  return ChunkStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), chunks, dim);
}

TensorStorageInfoPtrList ChunkCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>()) {
    MS_LOG(EXCEPTION) << "For '" << prim->name()
                      << ", first inputs must be a Tensor, but got: " << inputs[kInputIndex0]->ToString();
  }

  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto chunks = GetValue<int64_t>(inputs[kInputIndex1]);
  auto dim = GetValue<int64_t>(inputs[kInputIndex2]);
  return ChunkBasicTypeCalc(input_tensor, chunks, dim);
}

REG_TUPLE_OUT_VIEW_STRIDES_CALC_FUN(Chunk, ChunkCalc);
}  // namespace mindspore::ops
