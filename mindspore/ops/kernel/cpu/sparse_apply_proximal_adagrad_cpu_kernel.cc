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

#include "kernel/cpu/sparse_apply_proximal_adagrad_cpu_kernel.h"
#include <memory>
#include <map>
#include <utility>
#include "common/common_utils.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_proximal_adagrad_cpu {
using namespace sparse_optimizer_cpu;
namespace {
constexpr size_t kSparseApplyProximalAdagradInputsNum = 7;
constexpr size_t kSparseApplyProximalAdagradWorkspaceSize = 4;
constexpr char kKernelName[] = "SparseApplyProximalAdagrad";
constexpr size_t kVarIndex = 0;
constexpr size_t kAccIndex = 1;
constexpr size_t kLRIndex = 2;
constexpr size_t kL1Index = 3;
constexpr size_t kL2Index = 4;
constexpr size_t kGradIndex = 5;
constexpr size_t kIndicesIndex = 6;
constexpr size_t kWorkSpaceIndex0 = 0;
constexpr size_t kWorkSpaceIndex1 = 1;
constexpr size_t kWorkSpaceIndex2 = 2;
constexpr size_t kWorkSpaceIndex3 = 3;

using KernelRunFunc = SparseApplyProximalAdagradCpuKernelMod::KernelRunFunc;

template <typename T>
void ComputeProximalAdagrad(MultiThreadComputeParams<T> *input_params, size_t start, size_t end) {
  MS_EXCEPTION_IF_NULL(input_params);
  auto var = input_params->var_;
  auto accum = input_params->accum_;
  const auto lr = input_params->lr_;
  const auto l1 = input_params->l1_;
  const auto l2 = input_params->l2_;
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
      accum[j] += summed_grad * summed_grad;
      auto learning_rate = lr * (1 / std::sqrt(accum[j]));
      auto prox_v = var[j];
      prox_v -= summed_grad * learning_rate;
      if (l1 > 0) {
        var[j] = Sign(prox_v) * std::fmax(std::fabs(prox_v) - learning_rate * l1, static_cast<float>(0.0)) /
                 (1 + l2 * learning_rate);
      } else {
        var[j] = prox_v / (1 + l2 * learning_rate);
      }
    }
  }
}
}  // namespace

template <typename T>
void SparseApplyProximalAdagradCpuKernelMod::InitWorkspaceSize() {
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
  (void)workspace_size_list_.emplace_back(indices_size_ * var_outer_dim_size_ * sizeof(float));
  (void)workspace_size_list_.emplace_back(indices_size_ * sizeof(T));
}

bool SparseApplyProximalAdagradCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kSparseApplyProximalAdagradInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input size must be " << kSparseApplyProximalAdagradInputsNum
                  << ", but got " << inputs.size();
    return false;
  }
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

void SparseApplyProximalAdagradCpuKernelMod::ResetResource() noexcept {
  output_size_list_.clear();
  workspace_size_list_.clear();
  indices_data_type_ = kNumberTypeInt32;
  indices_size_ = 0;
  var_first_dim_size_ = 0;
  var_outer_dim_size_ = 1;
}

int SparseApplyProximalAdagradCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  ShapeVector var_shape = inputs[kVarIndex]->GetShapeVector();
  ShapeVector accum_shape = inputs[kAccIndex]->GetShapeVector();
  ShapeVector lr_shape = inputs[kLRIndex]->GetShapeVector();
  ShapeVector l1_shape = inputs[kL1Index]->GetShapeVector();
  ShapeVector l2_shape = inputs[kL2Index]->GetShapeVector();
  ShapeVector grad_shape = inputs[kGradIndex]->GetShapeVector();
  ShapeVector indices_shape = inputs[kIndicesIndex]->GetShapeVector();
  if (var_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'var' must be at least 1-D, but got scalar or None.";
    return KRET_RESIZE_FAILED;
  }
  if (!IsSameShape(var_shape, accum_shape)) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the shape of 'accum' must be the same as the shape of 'var', "
                     "but got the shape of 'accum': "
                  << accum_shape << " and the shape of 'var': " << var_shape;
    return KRET_RESIZE_FAILED;
  }
  if (var_shape.size() != grad_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'grad' must be the same as the dimension of "
                     "'var', but got the dimension of 'grad': "
                  << grad_shape.size() << " and the dimension of 'var': " << var_shape.size() << ".";
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
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the 'indices' must be a 1-D vector, but got "
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
  if (!lr_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'lr' must be a scalar,and dimension of 'lr' must be 0,but got the dimension of 'lr': "
                  << lr_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l1_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l1' must be a scalar,and dimension of 'l1' must be 0,but got the dimension of 'l1': "
                  << l1_shape;
    return KRET_RESIZE_FAILED;
  }
  if (!l2_shape.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', 'l2' must be a scalar,and dimension of 'l2' must be 0,but got the dimension of 'l2': "
                  << l2_shape;
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

const std::vector<std::pair<KernelAttr, KernelRunFunc>> &SparseApplyProximalAdagradCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &SparseApplyProximalAdagradCpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutInRef(0, 0),
     &SparseApplyProximalAdagradCpuKernelMod::LaunchKernel<int64_t>}};
  return func_list;
}

template <typename T>
bool SparseApplyProximalAdagradCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                                          const std::vector<kernel::KernelTensor *> &workspace,
                                                          const std::vector<kernel::KernelTensor *> &) const {
  auto var = reinterpret_cast<float *>(inputs[kVarIndex]->device_ptr());
  auto accum = reinterpret_cast<float *>(inputs[kAccIndex]->device_ptr());
  auto lr = reinterpret_cast<float *>(inputs[kLRIndex]->device_ptr())[0];
  auto l1 = reinterpret_cast<float *>(inputs[kL1Index]->device_ptr())[0];
  auto l2 = reinterpret_cast<float *>(inputs[kL2Index]->device_ptr())[0];
  auto grad = reinterpret_cast<float *>(inputs[kGradIndex]->device_ptr());
  auto indices = reinterpret_cast<T *>(inputs[kIndicesIndex]->device_ptr());
  auto new_grad = reinterpret_cast<float *>(workspace[kWorkSpaceIndex0]->device_ptr());
  auto new_indices = reinterpret_cast<T *>(workspace[kWorkSpaceIndex1]->device_ptr());
  auto workspace_grad = reinterpret_cast<float *>(workspace[kWorkSpaceIndex2]->device_ptr());
  auto workspace_indices = reinterpret_cast<T *>(workspace[kWorkSpaceIndex3]->device_ptr());

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

  MultiThreadComputeParams<T> input_params;
  input_params.var_ = var;
  input_params.accum_ = accum;
  input_params.lr_ = lr;
  input_params.l1_ = l1;
  input_params.l2_ = l2;
  input_params.sparse_grad_ = unique_sparse_grad;
  input_params.var_first_dim_size_ = var_first_dim_size_;
  input_params.var_outer_dim_size_ = var_outer_dim_size_;
  MultiThreadCompute<T>(ComputeProximalAdagrad<T>, &input_params, unique_sparse_grad.indices_size_);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FusedSparseProximalAdagrad, SparseApplyProximalAdagradCpuKernelMod);
}  // namespace sparse_apply_proximal_adagrad_cpu
}  // namespace kernel
}  // namespace mindspore
