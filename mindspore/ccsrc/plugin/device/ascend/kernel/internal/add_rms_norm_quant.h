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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_ADD_RMS_NORM_QUANT_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_ADD_RMS_NORM_QUANT_H_

#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/internal_kernel_mod.h"
#include "include/internal.h"

namespace mindspore {
namespace kernel {
class InternalAddRmsNormQuant : public InternalKernelMod {
 public:
  InternalAddRmsNormQuant() : InternalKernelMod() {}
  ~InternalAddRmsNormQuant() = default;

 protected:
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override;
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;

 private:
  internal::TensorFormat output_format_;
  bool need_rms_norm_out_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_ADD_RMS_NORM_QUANT_H_
