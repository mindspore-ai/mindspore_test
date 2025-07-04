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

#include "kernel/cpu/sparse_apply_adagrad_da_cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <memory>
#include <map>

#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_adagrad_da_cpu {
using namespace sparse_optimizer_cpu;
namespace {
constexpr size_t kSparseApplyAdagradDAInputsNum = 9;
constexpr size_t kSparseApplyAdagradDAOutputsNum = 1;

using KernelRunFunc = SparseApplyAdagradDACpuKernelMod::KernelRunFunc;

#define SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10) \
  KernelAttr()                                                                      \
    .AddInputAttr(kNumberType##t1)                                                  \
    .AddInputAttr(kNumberType##t2)                                                  \
    .AddInputAttr(kNumberType##t3)                                                  \
    .AddInputAttr(kNumberType##t4)                                                  \
    .AddInputAttr(kNumberType##t5)                                                  \
    .AddInputAttr(kNumberType##t6)                                                  \
    .AddInputAttr(kNumberType##t7)                                                  \
    .AddInputAttr(kNumberType##t8)                                                  \
    .AddInputAttr(kNumberType##t9)                                                  \
    .AddOutputAttr(kNumberType##t10)
}  // namespace

bool SparseApplyAdagradDACpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyAdagradDAInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyAdagradDAInputsNum
                  << ", but got " << inputs.size();
    return false;
  }
  if (outputs.size() != kSparseApplyAdagradDAOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be " << kSparseApplyAdagradDAOutputsNum
                  << ", but got " << outputs.size();
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyAdagradDACpuKernelMod::ResetResource() noexcept {
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyAdagradDACpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  enum input_index : size_t { Var_no, Ga_no, Gs_no, Grad_no, Indices_no, Lr_no, L1_no, L2_no, Global_step_no };
  ShapeVector var_shape = inputs[Var_no]->GetShapeVector();
  ShapeVector grad_accum_shape = inputs[Ga_no]->GetShapeVector();
  ShapeVector grad_square_accum_shape = inputs[Gs_no]->GetShapeVector();
  ShapeVector grad_shape = inputs[Grad_no]->GetShapeVector();
  ShapeVector indices_shape = inputs[Indices_no]->GetShapeVector();
  ShapeVector lr_shape = inputs[Lr_no]->GetShapeVector();
  ShapeVector l1_shape = inputs[L1_no]->GetShapeVector();
  ShapeVector l2_shape = inputs[L2_no]->GetShapeVector();
  ShapeVector global_step_shape = inputs[Global_step_no]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var must be at least 1D.";
  } else {
    var_first_dim_size_ = LongToSize(var_shape[kIndex0]);
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, rank(grad) should be same as rank(var), but got rank(grad): "
                      << grad_shape.size() << ", rank(var): " << var_shape.size() << ".";
  }
  if (!IsSameShape(var_shape, grad_accum_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad_accum should have the same shape.";
  }
  if (!IsSameShape(var_shape, grad_square_accum_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad_square_accum shape should have the same shape.";
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, var and grad should have the same shape size.";
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, the shape of var and grad must equal in dimension " << i << ".";
    }
    var_outer_dim_size_ *= LongToSize(var_shape[i]);
  }
  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, indices must be 1D, but got " << indices_shape.size() << ".";
  }
  indices_size_ = LongToSize(indices_shape[0]);
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(EXCEPTION)
      << "For SparseApplyAdagradDA, grad.shape[0] must be equal to indices.shape[0], but got grad_shape[0]: "
      << grad_shape[0] << ", indices_shape[0]: " << indices_size_ << ".";
  }
  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, lr is not a scalar, got shape: " << lr_shape << ".";
  }
  if (!l1_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, l1 is not a scalar, got shape: " << l1_shape << ".";
  }
  if (!l2_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, l2 is not a scalar, got shape: " << l2_shape << ".";
  }
  if (!global_step_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, global_step is not a scalar, got shape: " << global_step_shape
                      << ".";
  }
  return KRET_OK;
}

template <typename I, typename T>
bool SparseApplyAdagradDACpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                    const std::vector<kernel::KernelTensor *> &,
                                                    const std::vector<kernel::KernelTensor *> &outputs) const {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyAdagradDAInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyAdagradDAOutputsNum, kernel_name_);

  auto var = GetDeviceAddress<T>(inputs, kIndex0);
  auto ga = GetDeviceAddress<T>(inputs, kIndex1);
  auto da = GetDeviceAddress<T>(inputs, kIndex2);
  auto g = GetDeviceAddress<T>(inputs, kIndex3);
  auto indices = GetDeviceAddress<I>(inputs, kIndex4);
  auto lr_scalar = GetDeviceAddress<T>(inputs, kIndex5)[kIndex0];
  auto l1_scalar = GetDeviceAddress<T>(inputs, kIndex6)[kIndex0];
  auto l2_scalar = GetDeviceAddress<T>(inputs, kIndex7)[kIndex0];
  int64_t global_step_scalar_int64 = GetDeviceAddress<int64_t>(inputs, kIndex8)[kIndex0];
  T global_step_scalar = static_cast<T>(global_step_scalar_int64);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);

  auto gs_lr = global_step_scalar * lr_scalar;
  for (size_t i = 0; i < indices_size_; ++i) {
    I index = static_cast<I>(indices[i]);
    if (index < 0 || LongToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, values in indices should be [0, var.shape[0]), but got " << index
                        << ".";
    }
    size_t start_index = var_outer_dim_size_ * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      ga[j] = ga[j] + g[k];
      da[j] = da[j] + g[k] * g[k];
      if (l1_scalar > static_cast<T>(0.0)) {
        var[j] = static_cast<T>(-1.0) * static_cast<T>(Sign(static_cast<float>(ga[j]))) *
                 static_cast<T>(std::fmax(
                   static_cast<double>((static_cast<T>(std::fabs(static_cast<double>(ga[j]))) / global_step_scalar) -
                                       l1_scalar),
                   static_cast<double>(0.0))) /
                 (l2_scalar + static_cast<T>(std::sqrt(static_cast<double>(da[j]))) / gs_lr);
      } else {
        var[j] = static_cast<T>(-1.0) * (ga[j] / global_step_scalar) /
                 (l2_scalar + static_cast<T>(std::sqrt(static_cast<double>(da[j]))) / gs_lr);
      }
    }
  }
  size_t copy_size = var_first_dim_size_ * var_outer_dim_size_ * sizeof(T);
  auto ret = memcpy_s(output, copy_size, var, copy_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For SparseApplyAdagradDA, memcpy_s error, errorno" << ret << ".";
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyAdagradDACpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int8, Int8, Int8, Int8, Int32, Int8, Int8, Int8, Int64, Int8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int16, Int16, Int16, Int16, Int32, Int16, Int16, Int16, Int64, Int16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int64, Int32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int64, Int64, Int64, Int64, Int32, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float16, Float16, Float16, Float16, Int32, Float16, Float16, Float16, Int64,
                                        Float16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, float16>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float32, Float32, Float32, Float32, Int32, Float32, Float32, Float32, Int64,
                                        Float32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, float>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float64, Float64, Float64, Float64, Int32, Float64, Float64, Float64, Int64,
                                        Float64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int32_t, double>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int8, Int8, Int8, Int8, Int64, Int8, Int8, Int8, Int64, Int8),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int16, Int16, Int16, Int16, Int64, Int16, Int16, Int16, Int64, Int16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int32, Int32, Int32, Int32, Int64, Int32, Int32, Int32, Int64, Int32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float16, Float16, Float16, Float16, Int64, Float16, Float16, Float16, Int64,
                                        Float16),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, float16>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float32, Float32, Float32, Float32, Int64, Float32, Float32, Float32, Int64,
                                        Float32),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, float>},
    {SPARSE_APPLY_ADAGRAD_DA_ADD_KERNEL(Float64, Float64, Float64, Float64, Int64, Float64, Float64, Float64, Int64,
                                        Float64),
     &SparseApplyAdagradDACpuKernelMod::LaunchKernel<int64_t, double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyAdagradDA, SparseApplyAdagradDACpuKernelMod);
}  // namespace sparse_apply_adagrad_da_cpu
}  // namespace kernel
}  // namespace mindspore
