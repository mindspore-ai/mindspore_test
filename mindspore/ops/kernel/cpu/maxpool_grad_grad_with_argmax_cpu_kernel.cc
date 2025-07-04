/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/maxpool_grad_grad_with_argmax_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include <unordered_map>
#include "utils/profile.h"
#include "common/common_utils.h"

namespace mindspore {
namespace kernel {
namespace maxpool_grad_grad_with_argmax_cpu {
namespace {
constexpr size_t kMaxPoolGradGradWithArgmaxInputsNum = 3;
constexpr size_t kMaxPoolGradGradWithArgmaxOutputsNum = 1;
constexpr size_t kArgmaxIndex = 2;
}  // namespace

bool MaxPoolGradGradWithArgmaxCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kMaxPoolGradGradWithArgmaxInputsNum || outputs.size() != kMaxPoolGradGradWithArgmaxOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kMaxPoolGradGradWithArgmaxInputsNum
                  << " and " << kMaxPoolGradGradWithArgmaxOutputsNum << ", but get " << inputs.size() << " and "
                  << outputs.size();
    return false;
  }
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int MaxPoolGradGradWithArgmaxCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  auto in_shapes = inputs[0]->GetShapeVector();
  auto out_shapes = outputs[0]->GetShapeVector();
  output_elements_ = std::accumulate(out_shapes.begin(), out_shapes.end(), 1, std::multiplies<size_t>());
  input_batch_stride_ = std::accumulate(in_shapes.begin() + 1, in_shapes.end(), 1, std::multiplies<size_t>());
  output_batch_stride_ = std::accumulate(out_shapes.begin() + 1, out_shapes.end(), 1, std::multiplies<size_t>());
  return KRET_OK;
}

template <typename T, typename I>
bool MaxPoolGradGradWithArgmaxCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &,
                                                         const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMaxPoolGradGradWithArgmaxInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMaxPoolGradGradWithArgmaxOutputsNum, kernel_name_);

  T *grad = GetDeviceAddress<T>(inputs, kIndex1);
  I *argmax = GetDeviceAddress<I>(inputs, kArgmaxIndex);
  T *out = GetDeviceAddress<T>(outputs, kIndex0);

  auto task = [this, grad, argmax, out](size_t start, size_t end) {
    for (size_t pos = start; pos < end; pos++) {
      const int pos_n = pos / this->output_batch_stride_;
      out[pos] = grad[pos_n * this->input_batch_stride_ + argmax[pos]];
    }
  };
  ParallelLaunchAutoSearch(task, output_elements_, this, &parallel_search_info_, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, MaxPoolGradGradWithArgmaxCpuKernelMod::KernelRunFunc>>
  &MaxPoolGradGradWithArgmaxCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, MaxPoolGradGradWithArgmaxCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradGradWithArgmaxCpuKernelMod::LaunchKernel<float, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &MaxPoolGradGradWithArgmaxCpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradGradWithArgmaxCpuKernelMod::LaunchKernel<float16, int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &MaxPoolGradGradWithArgmaxCpuKernelMod::LaunchKernel<float16, int64_t>},
  };
  return func_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MaxPoolGradGradWithArgmax, MaxPoolGradGradWithArgmaxCpuKernelMod);
}  // namespace maxpool_grad_grad_with_argmax_cpu
}  // namespace kernel
}  // namespace mindspore
