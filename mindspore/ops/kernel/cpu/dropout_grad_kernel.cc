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

#include "kernel/cpu/dropout_grad_kernel.h"

#include <functional>
#include <map>
#include <utility>
#include <vector>
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

#include "kernel/cpu/nnacl/fp32_grad/dropout_grad.h"
#include "mindspore/ops/infer/grad/dropout_grad.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kDropoutGradInputsNum = 2;
constexpr size_t kDropoutGradOutputsNum = 1;
template <typename T>
CTask DoDropOutGrad(const T *input_addr, const T *mask_addr, T *output_addr, float keep_prob) {
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(mask_addr);
  MS_EXCEPTION_IF_NULL(output_addr);
  T scale = static_cast<T>(1.f / keep_prob);
  auto task = [input_addr, mask_addr, output_addr, scale](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_addr[i] = input_addr[i] * mask_addr[i] * scale;
    }
  };
  return task;
}

template <>
CTask DoDropOutGrad<float16>(const float16 *input_addr, const float16 *mask_addr, float16 *output_addr,
                             float keep_prob) {
  MS_EXCEPTION_IF_NULL(input_addr);
  MS_EXCEPTION_IF_NULL(mask_addr);
  MS_EXCEPTION_IF_NULL(output_addr);
  float scale = 1.f / keep_prob;
  auto task = [input_addr, mask_addr, output_addr, scale](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output_addr[i] = mask_addr[i] * static_cast<float16>(static_cast<float>(input_addr[i]) * scale);
    }
  };
  return task;
}
}  // namespace

using FuncVec = const std::vector<std::pair<KernelAttr, DropoutGradBwdCpuKernelMod::KernelRunFunc>>;

bool DropoutGradBwdCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kDropoutGradInputsNum || outputs.size() != kDropoutGradOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output tensor number must be " << kDropoutGradInputsNum
                  << " and " << kDropoutGradOutputsNum << ", but got " << inputs.size() << " and " << outputs.size();
    return false;
  }
  keep_prob_ = GetValue<float>(primitive_->GetAttr(ops::kKeepProb));
  if (keep_prob_ <= 0.0 || keep_prob_ > 1.0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << ", the 'keep_prob' must be in (0.0, 1.0], but got " << keep_prob_;
  }
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int DropoutGradBwdCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto dy_shape = Convert2SizeTClipNeg(inputs[kIndex0]->GetShapeVector());
  auto mask_shape = Convert2SizeTClipNeg(inputs[kIndex1]->GetShapeVector());
  if (dy_shape.size() != mask_shape.size()) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << ", the dimension of 'input' and 'input_mask' must be the same, "
                         "but got the dimension of 'input': "
                      << dy_shape.size() << ", and the dimension of 'input_mask': " << mask_shape.size();
  }

  num_count_ = std::accumulate(dy_shape.begin(), dy_shape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
  return static_cast<int>(KRET_OK);
}

template <typename T>
bool DropoutGradBwdCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &,
                                              const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDropoutGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDropoutGradOutputsNum, kernel_name_);

  T *output = GetDeviceAddress<T>(outputs, kIndex0);
  const T *input = GetDeviceAddress<T>(inputs, kIndex0);
  const T *mask = GetDeviceAddress<T>(inputs, kIndex1);
  auto task = DoDropOutGrad<T>(input, mask, output, keep_prob_);
  ParallelLaunchAutoSearch(task, num_count_, this, &parallel_search_info_);
  return true;
}

FuncVec &DropoutGradBwdCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, DropoutGradBwdCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &DropoutGradBwdCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &DropoutGradBwdCpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
     &DropoutGradBwdCpuKernelMod::LaunchKernel<double>},
  };
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, DropoutGrad, DropoutGradBwdCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
