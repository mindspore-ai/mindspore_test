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

#include "kernel/ascend/internal/pyboost/reshape_and_cache.h"

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr ReshapeAndCache::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                      const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateReshapeAndCacheOp(inputs, outputs, internal::kInternalReshapeAndCacheOpName);
}

void ReshapeAndCache::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                           const uint64_t &tiling_key, const TensorPtr &key, const std::optional<TensorPtr> &value,
                           const std::optional<TensorPtr> &key_cache, const std::optional<TensorPtr> &value_cache,
                           const std::optional<TensorPtr> &slot_mapping) {
  TensorPtrList inputs = {
    key, value.has_value() ? value.value() : nullptr, key_cache.has_value() ? key_cache.value() : nullptr,
    value_cache.has_value() ? value_cache.value() : nullptr, slot_mapping.has_value() ? slot_mapping.value() : nullptr};
  TensorPtrList outputs;
  TransInternalShapes(inputs, outputs);
  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(ReshapeAndCache, ReshapeAndCache);
}  // namespace kernel
}  // namespace mindspore
