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

#include "kernel/ascend/pyboost/customize/take_along_dim.h"
#include <algorithm>
#include <unordered_map>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<ValuePtr> infer_size(const ShapeVector &shape1, const ShapeVector &shape2) {
  auto dimsA = SizeToLong(shape1.size());
  auto dimsB = SizeToLong(shape2.size());
  auto ndim = dimsA > dimsB ? dimsA : dimsB;
  std::vector<ValuePtr> out_shape(ndim);
  for (auto i = ndim - 1; i >= 0; --i) {
    auto offset = ndim - 1 - i;
    auto dimA = dimsA - 1 - offset;
    auto dimB = dimsB - 1 - offset;
    auto sizeA = (dimA >= 0) ? shape1[dimA] : 1;
    auto sizeB = (dimB >= 0) ? shape2[dimB] : 1;
    if (sizeA != sizeB && sizeA != 1 && sizeB != 1) {
      MS_EXCEPTION(TypeError) << "For infer_size"
                              << ", the size of 'shape1' (" << sizeA << ") must match the size of 'shape2' (" << sizeB
                              << ") at non-singleton dimension " << i;
    }
    out_shape[i] = sizeA == 1 ? std::make_shared<Int64Imm>(sizeB) : std::make_shared<Int64Imm>(sizeA);
  }
  return out_shape;
}
}  // namespace
tensor::BaseTensorPtr TakeAlongDimAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const BaseTensorPtr &input_tensor, const BaseTensorPtr &indices,
                                                  const std::optional<Int64ImmPtr> &dim) {
  MS_LOG(DEBUG) << "TakeAloneDim Launch start";
  BaseTensorPtr output;
  if (dim.has_value()) {
    auto input_dim = SizeToLong(input_tensor->shape().size());
    auto indices_dim = SizeToLong(indices->shape().size());
    int64_t dim_imm = GetValue<int64_t>(dim.value());
    if (input_dim != indices_dim) {
      MS_EXCEPTION(TypeError) << "For TakeAloneDim"
                              << ", the dim of 'input' should be same as 'indices', but got 'input' with dim "
                              << input_dim << " and 'indices' with dim " << indices_dim << ".";
    }
    TypeId indices_type_id = static_cast<TypeId>(indices->data_type_c());
    if (indices_type_id != TypeId::kNumberTypeInt64) {
      MS_EXCEPTION(TypeError) << "For TakeAloneDim"
                              << ", the type of 'indices' must be Int64, but got 'indices' with type "
                              << TypeIdToString(indices_type_id);
    }
    dim_imm = dim_imm < 0 ? dim_imm + input_dim : dim_imm;
    auto input_shape = input_tensor->shape();
    input_shape[dim_imm] = indices->shape()[dim_imm];
    auto broadcast_shape = infer_size(input_shape, indices->shape());
    auto indices_broadcasted = broadcast_to(indices, std::make_shared<ValueTuple>(broadcast_shape));

    auto indices_shape = indices->shape();
    indices_shape[dim_imm] = input_tensor->shape()[dim_imm];
    broadcast_shape = infer_size(indices_shape, input_tensor->shape());
    auto input_broadcasted = broadcast_to(input_tensor, std::make_shared<ValueTuple>(broadcast_shape));
    output = gather_d(input_broadcasted, std::make_shared<Int64Imm>(dim_imm), indices_broadcasted);
  } else {
    std::vector<ValuePtr> negative_one{std::make_shared<Int64Imm>(-1)};
    output = reshape(input_tensor, std::make_shared<ValueTuple>(negative_one));
    output =
      gather_d(output, std::make_shared<Int64Imm>(0), reshape(indices, std::make_shared<ValueTuple>(negative_one)));
  }

  op->set_outputs({output});
  MS_LOG(DEBUG) << "TakeAloneDim Launch end";
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
