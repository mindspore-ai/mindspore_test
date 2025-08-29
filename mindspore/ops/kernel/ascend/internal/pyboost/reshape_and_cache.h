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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_RESHAPE_AND_CACHE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_RESHAPE_AND_CACHE_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class ReshapeAndCache : public InternalKernelInfo {
 public:
  explicit ReshapeAndCache(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~ReshapeAndCache() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const TensorPtr &key, const std::optional<TensorPtr> &value, const std::optional<TensorPtr> &key_cache,
            const std::optional<TensorPtr> &value_cache, const std::optional<TensorPtr> &slot_mapping);

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_RESHAPE_AND_CACHE_H_
