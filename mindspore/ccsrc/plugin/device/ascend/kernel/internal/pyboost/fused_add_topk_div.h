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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_FUSED_ADD_TOPK_DIV_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_FUSED_ADD_TOPK_DIV_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class FusedAddTopKDiv : public InternalKernelInfo {
 public:
  explicit FusedAddTopKDiv(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~FusedAddTopKDiv() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const BaseTensorPtr &x, const BaseTensorPtr &add_num, const int64_t &group_num, const int64_t &group_topk,
            const int64_t &n, const int64_t &k, const int64_t &activate_type, const bool &is_norm, const float &scale,
            const std::optional<BaseTensorPtr> &mapping_num, const std::optional<BaseTensorPtr> &mapping_table,
            const bool &enable_expert_mapping);

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  internal::FusedAddTopkDivParam param_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_PYBOOST_FUSED_ADD_TOPK_DIV_H_
