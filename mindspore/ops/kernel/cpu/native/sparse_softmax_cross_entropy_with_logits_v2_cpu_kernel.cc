/**
 * Copyright 2022-2024 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/sparse_softmax_cross_entropy_with_logits_v2_cpu_kernel.h"
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <functional>
namespace mindspore {
namespace kernel {
namespace sparse_softmax_cross_entropy_with_logits_v2_cpu {
namespace {
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2InputNum{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2OutputNum{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape{2};
constexpr std::size_t kSparseSoftmaxCrossEntropyWithLogitsV2LabelsShape{1};
}  // namespace

bool SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                                             const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SparseSoftmaxCrossEntropyWithLogitsV2 does not support this kernel data type: "
                      << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(outputs[kIndex0]->dtype_id());
  return true;
}

int SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  features_shape_ = inputs.at(kIndex0)->GetShapeVector();
  labels_shape_ = inputs.at(kIndex1)->GetShapeVector();
  auto features_batch = features_shape_[kIndex0];
  auto labels_batch = labels_shape_[kIndex0];
  if (features_shape_.size() != kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape ||
      labels_shape_.size() != kSparseSoftmaxCrossEntropyWithLogitsV2LabelsShape) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input logits(features) shape " << features_shape_
                      << " must be same as [batch * classes] and the input labels shape " << labels_shape_
                      << " must be same as [batch].";
  }
  if (features_batch != labels_batch) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the input logits(features) batch " << features_batch
                      << " must be equal to the input label batch " << labels_batch;
  }
  // apply workspace
  features_length_ =
    std::accumulate(features_shape_.begin(), features_shape_.end(), int64_t(1), std::multiplies<int64_t>());
  labels_length_ = std::accumulate(labels_shape_.begin(), labels_shape_.end(), int64_t(1), std::multiplies<int64_t>());
  workspace_size_list_.push_back(sizeof(float) * labels_length_);
  workspace_size_list_.push_back(sizeof(float) * features_length_);
  workspace_size_list_.push_back(unit_size_ * labels_length_);
  return KRET_OK;
}

template <typename data_type, typename label_type>
bool SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel(
  const std::vector<kernel::KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &workspace,
  const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseSoftmaxCrossEntropyWithLogitsV2InputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseSoftmaxCrossEntropyWithLogitsV2OutputNum, kernel_name_);
  auto *features = static_cast<data_type *>(inputs[kIndex0]->device_ptr());
  auto *labels = static_cast<label_type *>(inputs[kIndex1]->device_ptr());
  auto *loss = static_cast<data_type *>(outputs[kIndex0]->device_ptr());
  auto *backprop = static_cast<data_type *>(outputs[kIndex1]->device_ptr());
  const size_t batch_size = labels_length_;
  const size_t classes_num = features_length_ / labels_length_;
  for (size_t index = 0; index < labels_length_; index++) {
    if (labels[index] >= SizeToInt(classes_num) || labels[index] < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the labels[" << index << "] = " << labels[index]
                        << " value is outside the valid range of [0, " << classes_num << ").";
      return false;
    }
  }

  auto dims_exp_sum = GetDeviceAddress<float>(workspace, kIndex0);
  auto bp_fp32 = GetDeviceAddress<float>(workspace, kIndex1);
  auto dims_maximum = GetDeviceAddress<data_type>(workspace, kIndex2);
  if (memset_s(dims_exp_sum, batch_size * sizeof(float), 0, batch_size * sizeof(float)) != EOK) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', memset dims_exp_sum failed!";
  }
  Eigen::TensorMap<Eigen::Tensor<data_type, kSparseSoftmaxCrossEntropyWithLogitsV2FeaturesShape>, Eigen::Aligned>
    logits(features, batch_size, classes_num);
  Eigen::TensorMap<Eigen::Tensor<float, 1>, Eigen::Aligned> dims_sum(dims_exp_sum, batch_size);
  Eigen::TensorMap<Eigen::Tensor<data_type, 1>, Eigen::Aligned> dims_max(dims_maximum, batch_size);
  Eigen::array<int, 1> axes{{1}};
  // compute softmax
  dims_max = logits.maximum(axes);
  const data_type constant_one(1.0);
  for (size_t index = 0, batch_idx = 0; index < features_length_; index++) {
    bp_fp32[index] = Eigen::numext::exp(static_cast<float>(features[index] - dims_max(batch_idx)));
    dims_exp_sum[batch_idx] += bp_fp32[index];
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  dims_sum = dims_sum.inverse();
  for (size_t index = 0, batch_idx = 0; index < features_length_; index++) {
    backprop[index] = static_cast<data_type>(bp_fp32[index] * dims_sum(batch_idx));
    if ((index + 1) % classes_num == 0) {
      batch_idx++;
    }
  }
  for (size_t index = 0, batch_base = 0; index < batch_size; ++index, batch_base += classes_num) {
    size_t offset = static_cast<size_t>(labels[index]);
    loss[index] = -Eigen::numext::log(backprop[batch_base + offset]);
    backprop[batch_base + offset] = backprop[batch_base + offset] - constant_one;
  }
  return true;
}

std::vector<
  std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::SparseSoftmaxCrossEntropyWithLogitsV2Func>>
  SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<Eigen::half, std::int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<Eigen::half, std::int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<float, std::int32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::LaunchKernel<float, std::int64_t>}};
std::vector<KernelAttr> SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSoftmaxCrossEntropyWithLogitsV2Func> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SparseSoftmaxCrossEntropyWithLogitsV2,
                      SparseSoftmaxCrossEntropyWithLogitsV2CpuKernelMod);
}  // namespace sparse_softmax_cross_entropy_with_logits_v2_cpu
}  // namespace kernel
}  // namespace mindspore
