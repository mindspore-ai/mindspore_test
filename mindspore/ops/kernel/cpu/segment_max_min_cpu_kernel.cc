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
#include <algorithm>
#include <complex>
#include "kernel/cpu/segment_max_min_cpu_kernel.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace kernel {
namespace segment_max_min_cpu {
namespace {
const size_t kSegmentsThreshold = 2 * 1024;
const size_t kDataSizeThreshold = 2 * 1024;
}  // namespace

template <typename T>
void ComputeFuncMax(void *output_addr, void *input_addr) {
  T *output_ptr = static_cast<T *>(output_addr);
  T *input_ptr = static_cast<T *>(input_addr);
  auto a = *output_ptr;
  auto b = *input_ptr;
  *output_ptr = a > b ? a : b;
}

template <typename T>
void ComputeFuncMin(void *output_addr, void *input_addr) {
  T *output_ptr = static_cast<T *>(output_addr);
  T *input_ptr = static_cast<T *>(input_addr);
  auto a = *output_ptr;
  auto b = *input_ptr;
  *output_ptr = a < b ? a : b;
}

template <typename T>
T SegmentMaxMinCPUKernelMod::GetInitValue() const {
  static const std::map<std::string, T> SegmentMaxMinInitValueMap{{prim::kPrimSegmentMax->name(), static_cast<T>(0.0)},
                                                                  {prim::kPrimSegmentMin->name(), static_cast<T>(0.0)}};
  if (SegmentMaxMinInitValueMap.find(kernel_name_) == SegmentMaxMinInitValueMap.end()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the current operator does not support this operation.";
  }
  T init_value = SegmentMaxMinInitValueMap.at(kernel_name_);
  return init_value;
}

template <typename T>
bool SegmentMaxMinCPUKernelMod::GetComputeFunc() {
  static const std::map<std::string, SegmentComputeFunc> ComputeFuncList = {
    {prim::kPrimSegmentMax->name(), ComputeFuncMax<T>}, {prim::kPrimSegmentMin->name(), ComputeFuncMin<T>}};
  if (ComputeFuncList.find(kernel_name_) == ComputeFuncList.end()) {
    MS_LOG(ERROR) << "Invalid '" << kernel_name_ << "'.";
  }
  compute_func_ = ComputeFuncList.at(kernel_name_);
  return true;
}

bool SegmentMaxMinCPUKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  input_x_dtype_ = inputs.at(kIndex0)->dtype_id();
  segment_ids_dtype_ = inputs.at(kIndex1)->dtype_id();
  output_dtype_ = outputs.at(kIndex0)->dtype_id();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(EXCEPTION) << "SegmentMax does not support this kernel data type: " << kernel_attr;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SegmentMaxMinCPUKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  if (auto ret = NativeCpuKernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_x_shape_ = inputs.at(kIndex0)->GetShapeVector();
  segment_ids_shape_ = inputs.at(kIndex1)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  input_x_num_ = SizeOf(input_x_shape_);
  segment_ids_num_ = SizeOf(segment_ids_shape_);
  output_num_ = SizeOf(output_shape_);
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, SegmentMaxMinCPUKernelMod::SegmentMaxMinFunc>> SegmentMaxMinCPUKernelMod::func_list_ =
  {{KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<float16, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<float, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<double, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt8),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int8_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int16_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int32_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int64_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint8_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint16_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint32_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint64_t, int32_t>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<float16, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<float, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<double, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt8),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int8_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int16_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int32_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<int64_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint8_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt16),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint16_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt32),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint32_t, int64_t>},
   {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt64),
    &SegmentMaxMinCPUKernelMod::LaunchKernel<uint64_t, int64_t>}};

std::vector<KernelAttr> SegmentMaxMinCPUKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SegmentMaxMinFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename T1, typename T2>
bool SegmentMaxMinCPUKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                             const std::vector<kernel::KernelTensor *> &,
                                             const std::vector<kernel::KernelTensor *> &outputs) {
  if (kernel_name_ == prim::kPrimSegmentMax->name() || kernel_name_ == prim::kPrimSegmentMin->name()) {
    if constexpr (std::is_same_v<T1, std::complex<float>>) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex64.";
    } else if constexpr (std::is_same_v<T1, std::complex<double>>) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x types can not be complex128.";
    }
  }
  if (auto ret = GetComputeFunc<T1>(); !ret) {
    return ret;
  }
  T1 init_value = GetInitValue<T1>();
  auto input_x_data_addr = static_cast<T1 *>(inputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(input_x_data_addr);
  auto segment_ids_data_addr = static_cast<T2 *>(inputs[1]->device_ptr());
  MS_EXCEPTION_IF_NULL(segment_ids_data_addr);
  auto output_data_addr = static_cast<T1 *>(outputs[0]->device_ptr());
  MS_EXCEPTION_IF_NULL(output_data_addr);
  std::vector<int64_t> segments = CPUKernelUtils::CalcSegmentIds(segment_ids_data_addr, segment_ids_num_);
  for (size_t i = 0; i < output_num_; ++i) {
    output_data_addr[i] = init_value;
  }
  if (input_x_shape_[0] == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input_x_shape_[0] can not be 0";
  }
  const size_t num_compare_per = input_x_num_ / LongToSize(input_x_shape_[0]);
  const size_t num_segments = segments.size();
  if (num_segments < kSegmentsThreshold) {
    for (size_t i = 0; i < num_segments; ++i) {
      const size_t count = static_cast<size_t>(segments[i]);
      int64_t count_no = 0;
      for (size_t j = 0; j < i; ++j) {
        count_no += static_cast<int64_t>(segments[j]);
      }
      size_t input_addr_base = LongToSize(count_no) * num_compare_per;
      auto task = [&](size_t start, size_t end) {
        for (size_t j = start; j < end; ++j) {
          size_t res_init_addr = input_addr_base + j;
          T1 res_value = input_x_data_addr[res_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = SizeToInt(res_init_addr + k * num_compare_per);
            compute_func_(static_cast<void *>(&res_value), static_cast<void *>(input_x_data_addr + cmp_addr));
          }
          output_data_addr[static_cast<size_t>(segment_ids_data_addr[LongToSize(count_no)]) * num_compare_per + j] =
            res_value;
        }
      };
      if (num_compare_per < kDataSizeThreshold) {
        task(0, num_compare_per);
      } else {
        CPUKernelUtils::ParallelFor(task, num_compare_per);
      }
    }
  } else {
    auto task = [&](size_t start, size_t end) {
      for (size_t i = start; i < end; ++i) {
        const size_t count = static_cast<size_t>(segments[i]);
        int64_t count_no = 0;
        for (size_t j = 0; j < i; ++j) {
          count_no += static_cast<int64_t>(segments[j]);
        }
        size_t input_addr_base = LongToSize(count_no) * num_compare_per;
        for (size_t j = 0; j < num_compare_per; ++j) {
          size_t res_init_addr = input_addr_base + j;
          T1 res_value = input_x_data_addr[res_init_addr];
          for (size_t k = 1; k < count; ++k) {
            int cmp_addr = SizeToInt(res_init_addr + k * num_compare_per);
            compute_func_(static_cast<void *>(&res_value), static_cast<void *>(input_x_data_addr + cmp_addr));
          }
          output_data_addr[static_cast<size_t>(segment_ids_data_addr[LongToSize(count_no)]) * num_compare_per + j] =
            res_value;
        }
      }
    };
    CPUKernelUtils::ParallelFor(task, num_segments);
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SegmentMin, SegmentMaxMinCPUKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, SegmentMax, SegmentMaxMinCPUKernelMod);
}  // namespace segment_max_min_cpu
}  // namespace kernel
}  // namespace mindspore
