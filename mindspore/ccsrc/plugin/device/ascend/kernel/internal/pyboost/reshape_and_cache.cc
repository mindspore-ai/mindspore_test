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

#include "plugin/device/ascend/kernel/internal/pyboost/reshape_and_cache.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr ReshapeAndCache::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                      const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateReshapeAndCacheOp(inputs, outputs, internal::kInternalReshapeAndCacheOpName);
}

void ReshapeAndCache::Call(const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &key,
                           const std::optional<BaseTensorPtr> &value, const std::optional<BaseTensorPtr> &key_cache,
                           const std::optional<BaseTensorPtr> &value_cache,
                           const std::optional<BaseTensorPtr> &slot_mapping) {
  std::vector<BaseTensorPtr> inputs = {
    key, value.has_value() ? value.value() : nullptr, key_cache.has_value() ? key_cache.value() : nullptr,
    value_cache.has_value() ? value_cache.value() : nullptr, slot_mapping.has_value() ? slot_mapping.value() : nullptr};
  std::vector<BaseTensorPtr> outputs;
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(ReshapeAndCache, internal::kInternalReshapeAndCacheOpName, ReshapeAndCache);
}  // namespace kernel
}  // namespace mindspore
