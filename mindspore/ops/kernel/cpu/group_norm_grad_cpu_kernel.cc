/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/group_norm_grad_cpu_kernel.h"
#include <algorithm>
#include <functional>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "include/common/thread_pool.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGroupNormGradInputsNum = 9;
constexpr size_t kGroupNormGradOutputsNum = 3;
constexpr size_t minGroupNormGradInputDim = 2;
}  // namespace

bool GroupNormGradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GroupNormGradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }

  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid input size!";
  }

  const auto &x_shape_vector = inputs[kIndex1]->GetShapeVector();
  batch_ = LongToSize(x_shape_vector[kIndex0]);
  num_channel_ = LongToSize(x_shape_vector[kIndex1]);
  HxW_ = (x_shape_vector.size() == minGroupNormGradInputDim)
           ? 1
           : std::accumulate(x_shape_vector.begin() + kIndex2, x_shape_vector.end(), 1, std::multiplies<int64_t>());
  num_groups_ = LongToSize(inputs[kIndex5]->GetValueWithCheck<int64_t>());
  inner_size_ = num_channel_ * LongToSize(HxW_) / num_groups_;

  const size_t dscale_shape_size = batch_ * num_channel_ * sizeof(double);
  const size_t dbias_shape_size = batch_ * num_channel_ * sizeof(double);

  workspace_size_list_.clear();
  workspace_size_list_ = {dscale_shape_size, dbias_shape_size};

  return ret;
}

bool GroupNormGradCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &workspace,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGroupNormGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGroupNormGradOutputsNum, kernel_name_);
  kernel_func_(this, inputs, workspace, outputs);
  return true;
}

template <typename T>
void GroupNormGradCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  auto *dy = reinterpret_cast<T *>(inputs[kIndex0]->device_ptr());
  auto *x = reinterpret_cast<T *>(inputs[kIndex1]->device_ptr());
  auto *mean = reinterpret_cast<T *>(inputs[kIndex2]->device_ptr());
  auto *rstd = reinterpret_cast<T *>(inputs[kIndex3]->device_ptr());
  auto *gamma = reinterpret_cast<T *>(inputs[kIndex4]->device_ptr());
  auto *dx = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  auto *d_gamma = reinterpret_cast<T *>(outputs[kIndex1]->device_ptr());
  auto *d_beta = reinterpret_cast<T *>(outputs[kIndex2]->device_ptr());
  auto *dscale = reinterpret_cast<double *>(workspace[kIndex0]->device_ptr());
  auto *dbias = reinterpret_cast<double *>(workspace[kIndex1]->device_ptr());

  for (size_t idx = 0; idx < batch_ * num_channel_; ++idx) {
    T ds_val = (T)0.0;
    T db_val = (T)0.0;
    for (size_t j = idx * LongToSize(HxW_); j < (idx + 1) * LongToSize(HxW_); ++j) {
      ds_val += dy[j] * x[j];
      db_val += dy[j];
    }
    dscale[idx] = static_cast<double>(ds_val);
    dbias[idx] = static_cast<double>(db_val);
  }

  for (size_t param_index = 0; param_index < num_channel_; ++param_index) {
    T dg = (T)0.0;
    T db = (T)0.0;
    for (size_t j = 0; j < LongToSize(batch_); ++j) {
      auto idx1 = j * LongToSize(num_channel_) + param_index;
      auto idx2 = static_cast<size_t>(std::floor(idx1 * num_groups_ / num_channel_));
      dg += (static_cast<T>(dscale[idx1]) - static_cast<T>(dbias[idx1]) * mean[idx2]) * rstd[idx2];
      db += static_cast<T>(dbias[idx1]);
    }
    d_gamma[param_index] = dg;
    d_beta[param_index] = db;
  }

  auto task = [this, &dy, &x, &mean, &rstd, &dx, &gamma](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      double sum1 = 0.0;
      double sum2 = 0.0;
      double sum3 = 0.0;
      for (size_t j = i * inner_size_; j < (i + 1) * inner_size_; ++j) {
        auto param_index = (j / LongToSize(HxW_)) % num_channel_;
        auto dxm = static_cast<double>(x[j]) - static_cast<double>(mean[i]);
        auto dyg = static_cast<double>(dy[j] * gamma[param_index]);
        sum1 += dyg * dxm;
        sum2 += dyg;
        sum3 += dxm;
      }
      sum1 *= -0.5 * std::pow(static_cast<double>(rstd[i]), 3.0);
      sum3 *= -2.0;

      auto inv_inner_size = 1.0 / inner_size_;
      auto dx3 = 2.0 * sum1 * inv_inner_size;
      auto dx4 = (-1.0 * static_cast<double>(rstd[i]) * sum2 + sum1 * sum3 * inv_inner_size) * inv_inner_size;
      for (size_t j = i * inner_size_; j < (i + 1) * inner_size_; ++j) {
        auto param_index = (j / LongToSize(HxW_)) % num_channel_;
        auto dx1 = static_cast<double>(dy[j] * gamma[param_index]);
        auto dx2 = static_cast<double>(x[j]) - static_cast<double>(mean[i]);
        dx[j] = static_cast<T>(dx1 * static_cast<double>(rstd[i]) + dx2 * dx3 + dx4);
      }
    }
  };
  auto outter_size = LongToSize(batch_ * num_groups_);
  ParallelLaunchAutoSearch(task, outter_size, this, &parallel_search_info_);
}

std::vector<std::pair<KernelAttr, GroupNormGradCpuKernelMod::KernelFunc>> GroupNormGradCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeTuple, kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GroupNormGradCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GroupNormGradCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GroupNormGradCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> GroupNormGradCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, GroupNormGrad, GroupNormGradCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
