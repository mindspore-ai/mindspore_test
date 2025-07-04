/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/isinf_cpu_kernel.h"
#include <cmath>
#include "abstract/utils.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace isinf_cpu {
namespace {
constexpr size_t kIsInfInputsNum = 1;
constexpr size_t kIsInfOutputsNum = 1;
}  // namespace

bool IsInfCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_dtype_ = inputs[kIndex0]->dtype_id();
  if (dtype_map_.find(input_dtype_) == dtype_map_.end()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of 'x' must be bool, int, float, or uint, but got: " << input_dtype_;
  }
  return true;
}

bool IsInfCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsInfInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsInfOutputsNum, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernelFloat16(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32 || input_dtype_ == kNumberTypeFloat) {
    LaunchKernelFloat<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernelFloat<double>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dtype of 'x' must be float, but got "
                      << TypeIdLabel(input_dtype_);
  }
  return true;
}

void IsInfCpuKernelMod::LaunchKernelFloat16(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &outputs) const {
  float16 *input = GetDeviceAddress<float16>(inputs, kIndex0);
  bool *output = GetDeviceAddress<bool>(outputs, kIndex0);

  size_t elem_num = inputs[0]->size() / sizeof(float16);

  for (size_t i = 0; i < elem_num; i++) {
    float temp_num = static_cast<float>(input[i]);
    output[i] = std::isinf(temp_num);
  }
}

template <typename T>
void IsInfCpuKernelMod::LaunchKernelFloat(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<kernel::KernelTensor *> &outputs) const {
  T *input = GetDeviceAddress<T>(inputs, kIndex0);
  bool *output = GetDeviceAddress<bool>(outputs, kIndex0);

  size_t elem_num = inputs[0]->size() / sizeof(T);

  for (size_t i = 0; i < elem_num; i++) {
    output[i] = std::isinf(input[i]);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IsInf, IsInfCpuKernelMod);
}  // namespace isinf_cpu
}  // namespace kernel
}  // namespace mindspore
