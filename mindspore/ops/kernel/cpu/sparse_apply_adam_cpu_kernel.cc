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

#include "kernel/cpu/sparse_apply_adam_cpu_kernel.h"
#include <limits>
#include <memory>
#include <map>
#include <utility>
#include "common/common_utils.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "infer/fused_sparse_adam.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_adam_cpu {
using namespace sparse_optimizer_cpu;
namespace {
// "var","m","v","beta1_power","beta2_power","lr","beta1","beta2","epsilon","grad","indices"
constexpr size_t kVarIndex = 0;
constexpr size_t kMIndex = 1;
constexpr size_t kVIndex = 2;
constexpr size_t kBeta1PowerIndex = 3;
constexpr size_t kBeta2Powerndex = 4;
constexpr size_t kLrIndex = 5;
constexpr size_t kBeta1Index = 6;
constexpr size_t kBeta2Index = 7;
constexpr size_t kEpsilonIndex = 8;
constexpr size_t kGradIndex = 9;
constexpr size_t kIndicesIndex = 10;
constexpr size_t kSparseApplyAdamInputsNum = 11;
constexpr size_t kSparseApplyAdamWorkspaceSize = 5;
constexpr char kKernelName[] = "SparseApplyAdam";
using KernelRunFunc = SparseApplyAdamCpuKernelMod::KernelRunFunc;

template <typename T>
void ComputeAdam(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto m = input_params->m_;
  auto m_t = input_params->m_t_;
  auto v = input_params->v_;
  const auto beta1 = input_params->beta1_;
  const auto beta2 = input_params->beta2_;
  const auto use_nesterov = input_params->use_nesterov_;
  const auto unique_sparse_grad = input_params->sparse_grad_;
  const auto var_first_dim_size = input_params->var_first_dim_size_;
  const auto var_outer_dim_size = input_params->var_outer_dim_size_;
  for (size_t i = start; i < end; ++i) {
    T index = unique_sparse_grad.indices_[i];
    if (index < 0 || LongToSize(index) >= var_first_dim_size) {
      MS_LOG(EXCEPTION) << "For '" << kKernelName << "', each element in 'indices' must be in range [0, "
                        << SizeToLong(var_first_dim_size) << "), but got " << index;
    }
    size_t start_index = var_outer_dim_size * static_cast<size_t>(index);
    size_t end_index = start_index + var_outer_dim_size;
    for (size_t j = start_index, k = var_outer_dim_size * i; j < end_index; ++j, ++k) {
      auto summed_grad = unique_sparse_grad.value_[k];
      m[j] += (1 - beta1) * summed_grad;
      v[j] += (1 - beta2) * summed_grad * summed_grad;
      if (use_nesterov) {
        m_t[j] = m[j] * beta1 + (1 - beta1) * summed_grad;
      }
    }
  }
}

template <typename T>
void ComputeMomentum(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto m = input_params->m_;
  auto v = input_params->v_;
  const auto beta1 = input_params->beta1_;
  const auto beta2 = input_params->beta2_;
  for (size_t i = start; i < end; ++i) {
    m[i] *= beta1;
    v[i] *= beta2;
  }
}

template <typename T>
void ComputeWeight(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto var = input_params->var_;
  const auto *m = input_params->m_;
  const auto *v = input_params->v_;
  const auto lr = input_params->lr_;
  const auto epsilon = input_params->epsilon_;
  for (size_t i = start; i < end; ++i) {
    var[i] -= lr * m[i] / (std::sqrt(v[i]) + epsilon);
  }
}
}  // namespace

template <typename T>
void SparseApplyAdamCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(var_first_dim_size_ * var_outer_dim_size_ * sizeof(float));
}

// Initialization for the kernel mod.
bool SparseApplyAdamCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyAdamInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyAdamInputsNum << ", but got "
                  << inputs.size();
    return false;
  }
  use_nesterov_ = GetValue<bool>(primitive_->GetAttr(ops::kUseNesterov));
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyAdamCpuKernelMod::ResetResource() noexcept {
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyAdamCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> var_shape = inputs[kVarIndex]->GetShapeVector();
  std::vector<int64_t> m_shape = inputs[kMIndex]->GetShapeVector();
  std::vector<int64_t> v_shape = inputs[kVIndex]->GetShapeVector();
  std::vector<int64_t> grad_shape = inputs[kGradIndex]->GetShapeVector();
  std::vector<int64_t> indices_shape = inputs[kIndicesIndex]->GetShapeVector();

  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, m_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'm' must be the same as the shape of 'var', but got the shape of 'm': " << m_shape
                  << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, v_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'v' must be the same as the shape of 'var', but got the shape of 'v': " << v_shape
                  << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'grad' must be the same as the dimension of "
                     "'var', but got the dimension of 'grad': "
                  << grad_shape.size() << " and the dimension of 'var': " << var_shape.size();
    return KRET_RESIZE_FAILED;
  }
  var_first_dim_size_ = LongToSize(var_shape[0]);
  for (size_t i = 1; i < var_shape.size(); ++i) {
    if (var_shape[i] != grad_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of 'var' and 'grad' must be equal in dimension i=" << i
                    << ", but got 'var_shape[i]': " << var_shape[i] << " and 'grad_shape[i]': " << grad_shape[i];
      return KRET_RESIZE_FAILED;
    }
    var_outer_dim_size_ *= LongToSize(var_shape[i]);
  }
  if (indices_shape.size() != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'indices' must be 1-D, but got "
                  << indices_shape.size() << "-D.";
    return KRET_RESIZE_FAILED;
  }
  indices_size_ = LongToSize(indices_shape[0]);
  if (grad_shape[0] != SizeToLong(indices_size_)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the first dimension value of 'grad' must be equal to "
                     "the first dimension value of 'indices', but got the first dimension value of 'grad': "
                  << grad_shape[0] << ", and the first dimension value of 'indices': " << indices_size_;
    return KRET_RESIZE_FAILED;
  }
  indices_data_type_ = inputs[kIndicesIndex]->dtype_id();
  if (indices_data_type_ == kNumberTypeInt32) {
    InitWorkspaceSize<int>();
  } else if (indices_data_type_ == kNumberTypeInt64) {
    InitWorkspaceSize<int64_t>();
  } else {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dtype of 'indices' must be int32 or int64, but got "
                  << TypeIdToType(indices_data_type_)->ToString();
    return KRET_RESIZE_FAILED;
  }
  return KRET_OK;
}

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyAdamCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &SparseApplyAdamCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &SparseApplyAdamCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}

template <typename T>
bool SparseApplyAdamCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                               const std::vector<kernel::KernelTensor *> &workspace,
                                               const std::vector<kernel::KernelTensor *> &) const {
  auto *var = GetDeviceAddress<float>(inputs, 0);
  auto *m = GetDeviceAddress<float>(inputs, 1);
  auto *v = GetDeviceAddress<float>(inputs, 2);
  auto beta1_power = GetDeviceAddress<float>(inputs, 3)[0];
  if (std::fabs(beta1_power - 1.0f) <= std::numeric_limits<float>::epsilon()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'beta1_power' can not be 1.";
    return false;
  }
  auto beta2_power = GetDeviceAddress<float>(inputs, 4)[0];
  auto lr = GetDeviceAddress<float>(inputs, 5)[0];
  auto beta1 = GetDeviceAddress<float>(inputs, 6)[0];
  auto beta2 = GetDeviceAddress<float>(inputs, 7)[0];
  auto epsilon = GetDeviceAddress<float>(inputs, 8)[0];
  auto *grad = GetDeviceAddress<float>(inputs, 9);
  auto *indices = GetDeviceAddress<T>(inputs, 10);
  auto *new_grad = GetDeviceAddress<float>(workspace, 0);
  auto *new_indices = GetDeviceAddress<T>(workspace, 1);
  auto *workspace_grad = GetDeviceAddress<float>(workspace, 2);
  auto *workspace_indices = GetDeviceAddress<T>(workspace, 3);
  auto *m_t = GetDeviceAddress<float>(workspace, 4);

  SparseGradient<T> unique_sparse_grad({new_grad, new_indices, indices_size_});
  SparseGradient<T> workspace_sparse_grad({workspace_grad, workspace_indices, indices_size_});
  SparseGradient<T> input_sparse_grad({grad, indices, indices_size_});
  ReduceSparseGradientParam<T> param;
  param.input_grad_ = &input_sparse_grad;
  param.workspace_grad_ = &workspace_sparse_grad;
  param.output_grad_ = &unique_sparse_grad;
  param.max_index_ = var_first_dim_size_;
  param.value_stride_ = var_outer_dim_size_;
  BucketReduceSparseGradient(param);

  size_t total_dim_size = var_first_dim_size_ * var_outer_dim_size_;
  lr = lr * std::sqrt(1 - beta2_power) / (1 - beta1_power);

  MultiThreadComputeParams<T> input_params;
  input_params.m_ = m;
  input_params.v_ = v;
  input_params.beta1_ = beta1;
  input_params.beta2_ = beta2;
  MultiThreadCompute<T>(ComputeMomentum<T>, &input_params, total_dim_size);
  input_params.m_t_ = m_t;
  input_params.use_nesterov_ = use_nesterov_;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;
  MultiThreadCompute<T>(ComputeAdam<T>, &input_params, unique_sparse_grad.indices_size_);

  if (use_nesterov_) {
    input_params.m_ = input_params.m_t_;
  }
  input_params.var_ = var;
  input_params.lr_ = lr;
  input_params.epsilon_ = epsilon;
  MultiThreadCompute<T>(ComputeWeight<T>, &input_params, total_dim_size);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FusedSparseAdam, SparseApplyAdamCpuKernelMod);
}  // namespace sparse_apply_adam_cpu
}  // namespace kernel
}  // namespace mindspore
