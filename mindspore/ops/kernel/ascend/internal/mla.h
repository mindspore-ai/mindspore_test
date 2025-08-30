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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_MLA_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_MLA_H_

#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/internal_kernel_mod.h"
#include "include/internal.h"

namespace mindspore {
namespace kernel {
class InternalMla : public InternalKernelMod {
 public:
  InternalMla() : InternalKernelMod() {}
  ~InternalMla() = default;

 protected:
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override;
  bool UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;

 private:
  bool created_flag_{false};
  internal::MLAParam param_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_KERNEL_INTERNAL_MLA_H_
