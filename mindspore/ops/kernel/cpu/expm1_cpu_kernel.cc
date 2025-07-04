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

#include "kernel/cpu/expm1_cpu_kernel.h"
#include <cmath>
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace expm1_cpu {
namespace {
constexpr size_t kExpm1InputsNum = 1;
constexpr size_t kExpm1OutputsNum = 1;
}  // namespace

bool Expm1CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  input_dtype_ = inputs[0]->dtype_id();
  if (input_dtype_ != kNumberTypeFloat16 && input_dtype_ != kNumberTypeFloat32 && input_dtype_ != kNumberTypeFloat64 &&
      input_dtype_ != kNumberTypeComplex64 && input_dtype_ != kNumberTypeComplex128) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dtype of input should be Float16, Float32, Float64, Complex64 or Complex128, but got: "
                  << input_dtype_;
    return false;
  }
  return true;
}

bool Expm1CpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kExpm1InputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kExpm1OutputsNum, kernel_name_);
  if (input_dtype_ == kNumberTypeFloat16) {
    LaunchKernel<float16>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat32) {
    LaunchKernel<float>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeFloat64) {
    LaunchKernel<double>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeComplex64) {
    LaunchKernel<std::complex<float>>(inputs, outputs);
  } else if (input_dtype_ == kNumberTypeComplex128) {
    LaunchKernel<std::complex<double>>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "', the dtype of input should be Float16, Float32, Float64, Complex64 or Complex128, but got: "
                      << TypeIdLabel(input_dtype_);
  }
  return true;
}

template <typename T>
void Expm1CpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) const {
  const auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  size_t elem_num = inputs[0]->size() / sizeof(T);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = exp(input[i]) - T(1);
  }
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Expm1, Expm1CpuKernelMod);
}  // namespace expm1_cpu
}  // namespace kernel
}  // namespace mindspore
