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

#include "kernel/cpu/sparse_apply_centered_rms_prop_cpu_kernel.h"

#include <algorithm>
#include <utility>
#include <map>

#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_centered_rms_prop_cpu {
using namespace sparse_optimizer_cpu;
namespace {
constexpr size_t kSparseApplyCenteredRMSPropInputsNum = 10;
constexpr size_t kSparseApplyCenteredRMSPropOutputsNum = 1;
using KernelRunFunc = SparseApplyCenteredRMSPropCpuKernelMod::KernelRunFunc;

#define SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11) \
  KernelAttr()                                                                                  \
    .AddInputAttr(kNumberType##t1)                                                              \
    .AddInputAttr(kNumberType##t2)                                                              \
    .AddInputAttr(kNumberType##t3)                                                              \
    .AddInputAttr(kNumberType##t4)                                                              \
    .AddInputAttr(kNumberType##t5)                                                              \
    .AddInputAttr(kNumberType##t6)                                                              \
    .AddInputAttr(kNumberType##t7)                                                              \
    .AddInputAttr(kNumberType##t8)                                                              \
    .AddInputAttr(kNumberType##t9)                                                              \
    .AddInputAttr(kNumberType##t10)                                                             \
    .AddOutputAttr(kNumberType##t11)
}  // namespace

bool SparseApplyCenteredRMSPropCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyCenteredRMSPropInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyCenteredRMSPropInputsNum
                  << ", but got " << inputs.size();
    return false;
  }
  if (outputs.size() != kSparseApplyCenteredRMSPropOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', output size must be " << kSparseApplyCenteredRMSPropOutputsNum
                  << ", but got " << outputs.size();
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyCenteredRMSPropCpuKernelMod::ResetResource() noexcept {
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyCenteredRMSPropCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  enum input_index : size_t {
    Var_no,
    Mg_no,
    Ms_no,
    Mom_no,
    Lr_no,
    Rho_no,
    Momentum_no,
    Epsilon_no,
    Grad_no,
    Indices_no
  };
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyCenteredRMSPropInputsNum, kernel_name_);
  auto var_shape = inputs[Var_no]->GetShapeVector();
  auto mg_shape = inputs[Mg_no]->GetShapeVector();
  auto ms_shape = inputs[Ms_no]->GetShapeVector();
  auto mom_shape = inputs[Mom_no]->GetShapeVector();
  auto lr_shape = inputs[Lr_no]->GetShapeVector();
  auto rho_shape = inputs[Rho_no]->GetShapeVector();
  auto momentum_shape = inputs[Momentum_no]->GetShapeVector();
  auto epsilon_shape = inputs[Epsilon_no]->GetShapeVector();
  auto grad_shape = inputs[Grad_no]->GetShapeVector();
  auto indices_shape = inputs[Indices_no]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, var must be at least 1D";
  } else {
    var_first_dim_size_ = LongToSize(var_shape[0]);
  }
  if (!IsSameShape(var_shape, mg_shape) && !IsSameShape(var_shape, ms_shape) && !IsSameShape(var_shape, mom_shape) &&
      !IsSameShape(var_shape, grad_shape)) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, var, mg, ms, mom and grad should have the same shape.";
  }
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, the shape of var and grad must equal in dimension " << i
                        << ".";
    }
    var_outer_dim_size_ *= LongToSize(var_shape[i]);
  }

  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, rank(grad) should be same as rank(var), but got rank(grad): "
                      << grad_shape.size() << ", rank(var): " << var_shape.size() << ".";
  }

  if (indices_shape.size() != 1) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, indices must be 1D, but got " << indices_shape.size() << "D.";
  }
  indices_size_ = LongToSize(indices_shape[0]);
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(EXCEPTION)
      << "For SparseApplyCenteredRMSProp, grad.shape[0] must be equal to indices.shape[0], but got grad_shape[0]: "
      << grad_shape[0] << " indices_shape[0]: " << indices_size_ << ".";
  }
  if (!lr_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, lr is not a scalar, got shape: " << lr_shape << ".";
  }
  if (!rho_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, rho is not a scalar, got shape: " << rho_shape << ".";
  }
  if (!momentum_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, momentum is not a scalar, got shape: " << momentum_shape
                      << ".";
  }
  if (!epsilon_shape.empty()) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, epsilon is not a scalar, got shape: " << epsilon_shape << ".";
  }
  return KRET_OK;
}

template <typename I, typename T>
bool SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                          const std::vector<kernel::KernelTensor *> &,
                                                          const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseApplyCenteredRMSPropInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseApplyCenteredRMSPropOutputsNum, kernel_name_);

  auto var = GetDeviceAddress<T>(inputs, kIndex0);
  auto mg = GetDeviceAddress<T>(inputs, kIndex1);
  auto ms = GetDeviceAddress<T>(inputs, kIndex2);
  auto mom = GetDeviceAddress<T>(inputs, kIndex3);
  auto lr_scalar = GetDeviceAddress<T>(inputs, kIndex4)[0];
  auto rho_scalar = GetDeviceAddress<T>(inputs, kIndex5)[0];
  auto momentum_scalar = GetDeviceAddress<T>(inputs, kIndex6)[0];
  auto epsilon_scalar = GetDeviceAddress<T>(inputs, kIndex7)[0];
  auto grad = GetDeviceAddress<T>(inputs, kIndex8);
  auto indices = GetDeviceAddress<I>(inputs, kIndex9);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  for (size_t i = 0; i < indices_size_; ++i) {
    I index = indices[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size_) {
      MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, values in indices should be [0, var.shape[0]), but got "
                        << index << ".";
    }
    size_t start_index = var_outer_dim_size_ * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size_;
    for (size_t j = start_index, k = var_outer_dim_size_ * i; j < end_index; ++j, ++k) {
      ms[j] = ms[j] * rho_scalar + grad[k] * grad[k] * (static_cast<T>(1) - rho_scalar);
      mg[j] = mg[j] * rho_scalar + grad[k] * (static_cast<T>(1) - rho_scalar);
      auto denom = ms[j] + epsilon_scalar - mg[j] * mg[j];
      mom[j] = mom[j] * momentum_scalar + (T)(1 / std::sqrt(static_cast<double>(denom))) * lr_scalar * grad[k];
      var[j] -= mom[j];
    }
  }
  size_t copy_size = var_first_dim_size_ * var_outer_dim_size_ * sizeof(T);
  auto ret = memcpy_s(output, copy_size, var, copy_size);
  if (ret != EOK) {
    MS_LOG(EXCEPTION) << "For SparseApplyCenteredRMSProp, memcpy_s error, errorno" << ret;
  }
  return true;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyCenteredRMSPropCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int32, Int8),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, int8_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int32,
                                               Int16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, int16_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32,
                                               Int32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int32,
                                               Int64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, Int32,
                                               UInt8),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, uint8_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16,
                                               Int32, UInt16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, uint16_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                               Int32, UInt32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, uint32_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                                               Int32, UInt64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, uint64_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16,
                                               Float16, Int32, Float16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, float16>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
                                               Float32, Int32, Float32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, float>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
                                               Float64, Int32, Float64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int32_t, double>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int8, Int32, Int8),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, int8_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int16, Int64,
                                               Int16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, int16_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int64,
                                               Int32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, int32_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64, Int64,
                                               Int64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, UInt8, Int64,
                                               UInt8),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, uint8_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16, UInt16,
                                               Int64, UInt16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, uint16_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32, UInt32,
                                               Int64, UInt32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, uint32_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64, UInt64,
                                               Int64, UInt64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, uint64_t>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float16, Float16, Float16, Float16, Float16, Float16, Float16, Float16,
                                               Float16, Int64, Float16),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, float16>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float32, Float32, Float32, Float32, Float32, Float32, Float32, Float32,
                                               Float32, Int64, Float32),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, float>},
    {SPARSE_APPLY_CENTERED_RMS_PROP_ADD_KERNEL(Float64, Float64, Float64, Float64, Float64, Float64, Float64, Float64,
                                               Float64, Int64, Float64),
     &SparseApplyCenteredRMSPropCpuKernelMod::LaunchKernel<int64_t, double>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseApplyCenteredRMSProp, SparseApplyCenteredRMSPropCpuKernelMod);
}  // namespace sparse_apply_centered_rms_prop_cpu
}  // namespace kernel
}  // namespace mindspore
