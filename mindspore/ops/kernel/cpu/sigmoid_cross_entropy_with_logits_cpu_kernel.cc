/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/sigmoid_cross_entropy_with_logits_cpu_kernel.h"
#include <map>
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace sigmoid_cross_entropy_with_logits_cpu {
namespace {
constexpr size_t kSigmoidCrossEntropyWithLogitsInputsNum = 2;
constexpr size_t kSigmoidCrossEntropyWithLogitsOutputsNum = 1;
}  // namespace

bool SigmoidCrossEntropyWithLogitsCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  return True;
}

int SigmoidCrossEntropyWithLogitsCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  dtype_ = inputs.at(0)->dtype_id();
  auto x_shape = inputs.at(0)->GetShapeVector();
  tensor_size_ = SizeOf(x_shape);
  return KRET_OK;
}

bool SigmoidCrossEntropyWithLogitsCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                                       const std::vector<kernel::KernelTensor *> &,
                                                       const std::vector<kernel::KernelTensor *> &outputs) {
  if (dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (dtype_ == kNumberTypeFloat32 || dtype_ == kNumberTypeFloat64) {
    LaunchKernel<float>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input must be float16, float32, or float64, but got " << TypeIdLabel(dtype_);
  }
  return true;
}

template <typename T>
void SigmoidCrossEntropyWithLogitsCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSigmoidCrossEntropyWithLogitsInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSigmoidCrossEntropyWithLogitsOutputsNum, kernel_name_);
  auto *logits_addr = GetDeviceAddress<T>(inputs, 0);
  auto *labels_addr = GetDeviceAddress<T>(inputs, 1);
  auto *output_addr = GetDeviceAddress<T>(outputs, 0);
  auto zero = static_cast<T>(0.0);
  auto one = static_cast<T>(1.0);
  auto two = static_cast<T>(2.0);
  for (uint64_t i = 0; i < tensor_size_; ++i) {
    if (logits_addr[i] >= zero) {
      output_addr[i] = static_cast<T>(log1p(static_cast<float>(exp(logits_addr[i] - two * logits_addr[i])))) -
                       logits_addr[i] * (labels_addr[i] - one);
    } else {
      output_addr[i] = static_cast<T>(log1p(static_cast<float>(exp(logits_addr[i])))) - logits_addr[i] * labels_addr[i];
    }
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SigmoidCrossEntropyWithLogits, SigmoidCrossEntropyWithLogitsCpuKernelMod);
}  // namespace sigmoid_cross_entropy_with_logits_cpu
}  // namespace kernel
}  // namespace mindspore
