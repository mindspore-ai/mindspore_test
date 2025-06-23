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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_MOE_GATING_GROUP_TOPK_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_MOE_GATING_GROUP_TOPK_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class MoeGatingGroupTopK : public InternalKernelInfo {
 public:
  explicit MoeGatingGroupTopK(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~MoeGatingGroupTopK() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const BaseTensorPtr &x_tensor, const std::optional<BaseTensorPtr> &bias_tensor, const int64_t &k,
            const int64_t &k_group, const int64_t &group_count, const int64_t &group_select_mode, const int64_t &renorm,
            const int64_t &norm_type, const bool &out_flag, const float &routed_scaling_factor, const float &eps);

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  internal::MoeGatingGroupTopKParam param_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_PYBOOST_MOE_GATING_GROUP_TOPK_H_
