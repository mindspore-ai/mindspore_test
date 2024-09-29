/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_FLASHATTENTIONSCORE_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_FLASHATTENTIONSCORE_

#include <vector>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"

namespace mindspore {
namespace kernel {
class InternalFlashAttentionScore : public InternalKernelMod {
 public:
  InternalFlashAttentionScore() : InternalKernelMod("FlashAttentionScore") {}
  ~InternalFlashAttentionScore() = default;

 protected:
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  internal::OpParamPtr CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs);
  void SetKVHead(internal::MixParam *op_param, int64_t head_num, int64_t input_layout);

 private:
  bool enable_internal_fa_{false};
  ShapeVector q_shape_;
  ShapeVector kv_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_FLASHATTENTIONSCORE_
