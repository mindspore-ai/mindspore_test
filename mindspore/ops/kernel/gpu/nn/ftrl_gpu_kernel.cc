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
#include <algorithm>
#include "kernel/gpu/nn/ftrl_gpu_kernel.h"
#include "abstract/utils.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "common/common_utils.h"

namespace mindspore {
namespace kernel {
bool FtrlGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 9;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

template <typename T>
bool FtrlGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_null_input_) {
    return true;
  }
  T *variable = GetDeviceAddress<T>(inputs, 0);
  T *accumulation = GetDeviceAddress<T>(inputs, 1);
  T *linear = GetDeviceAddress<T>(inputs, 2);
  T *gradient = GetDeviceAddress<T>(inputs, 3);
  T *learning_rate = GetDeviceAddress<T>(inputs, 4);
  T *l1_regularization = GetDeviceAddress<T>(inputs, 5);
  T *l2_regularization = GetDeviceAddress<T>(inputs, 6);
  T *learning_rate_power = GetDeviceAddress<T>(inputs, 7);
  auto status =
    ApplyFtrl(inputs[0]->size() / sizeof(T), gradient, learning_rate, l1_regularization, l2_regularization,
              learning_rate_power, variable, accumulation, linear, reinterpret_cast<cudaStream_t>(stream_ptr));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define FTRL_GPU_REG(MS_T, T)                                        \
  std::make_pair(KernelAttr()                                        \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(MS_T)                               \
                   .AddInputAttr(kObjectTypeNumber, kNumberTypeBool) \
                   .AddOutputAttr(MS_T),                             \
                 &FtrlGpuKernelMod::LaunchKernel<T>)

std::vector<std::pair<KernelAttr, FtrlGpuKernelMod::FtrlLaunchFunc>> FtrlGpuKernelMod::func_list_ = {
  FTRL_GPU_REG(kNumberTypeFloat16, half),
  FTRL_GPU_REG(kNumberTypeFloat32, float),
  FTRL_GPU_REG(kNumberTypeFloat64, double),
  FTRL_GPU_REG(kNumberTypeInt8, int8_t),
  FTRL_GPU_REG(kNumberTypeInt16, int16_t),
  FTRL_GPU_REG(kNumberTypeInt64, int64_t),
  FTRL_GPU_REG(kNumberTypeUInt8, uint8_t),
  FTRL_GPU_REG(kNumberTypeUInt16, uint16_t),
  FTRL_GPU_REG(kNumberTypeUInt32, uint32_t),
  FTRL_GPU_REG(kNumberTypeUInt64, uint64_t),
  FTRL_GPU_REG(kNumberTypeComplex64, utils::Complex<float>),
  FTRL_GPU_REG(kNumberTypeComplex128, utils::Complex<double>)};

std::vector<KernelAttr> FtrlGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, FtrlGpuKernelMod::FtrlLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ApplyFtrl, FtrlGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
