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

#include "kernel/cpu/native/scatter_value_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <functional>

#include "mindspore/ops/infer/ops_func_impl/tensor_scatter_elements.h"

namespace mindspore::kernel {
namespace scatter_value_cpu {
namespace {
template <class T>
struct ReductionAdd {
  void operator()(T *a, const T &b) const { (*a) += b; }
};

template <class T>
struct ReductionAssignment {
  void operator()(T *a, const T &b) const { (*a) = b; }
};
}  // namespace

bool ScatterValueCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ != kernel_type_) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }
  reduction_type_ = inputs[kIndex4]->GetValueWithCheck<Reduce>();
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ScatterValueCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  input_dims_ = input_shape.size();
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the dimension of 'input' should be greater than or equal to 1, but got " << input_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  auto indices_shape = inputs[kIndex2]->GetShapeVector();
  if (input_dims_ != indices_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input', 'index' should be same, but got "
                  << "input dims: " << input_dims_ << "; "
                  << "index dims: " << indices_shape.size() << ". ";
    return KRET_RESIZE_FAILED;
  }

  axis_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (axis_ < 0) {
    axis_ += static_cast<int64_t>(input_dims_);
  }

  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the 'dim' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape[i] < indices_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the indices dims should be less than input dims, but got indices dim is: "
                    << indices_shape[i] << " at axis: " << i << ", while input dim is:" << input_shape[i];
      return KRET_RESIZE_FAILED;
    }
  }

  input_axis_size_ = SizeToInt(input_shape[axis_]);
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  indices_total_num_ =
    std::accumulate(indices_shape.begin(), indices_shape.end(), size_t(1), std::multiplies<size_t>());

  // calculate indices_stride
  indices_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    indices_stride_[i - 1] = indices_stride_[i] * indices_shape[i];
  }

  // calculate output_stride
  output_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    output_stride_[i - 1] = output_stride_[i] * static_cast<size_t>(input_shape[i]);
  }
  return KRET_OK;
}

template <typename T, typename S, typename ReductionT>
bool ScatterValueCpuKernelMod::Scatter(const ReductionT &reduction_func, T *output, const S *indices, T src) {
  auto task = [reduction_func, output, indices, src, this](size_t start, size_t end) {
    for (size_t index = start; index < end; index++) {
      int remain = static_cast<int>(index);
      int output_offset = 0;
      for (size_t i = 0; i < this->input_dims_; ++i) {
        int output_dim_index = remain / this->indices_stride_[i];
        remain %= this->indices_stride_[i];
        if (i == static_cast<size_t>(this->axis_)) {
          output_dim_index = *(indices + index);
          if (output_dim_index >= this->input_axis_size_ || output_dim_index < -this->input_axis_size_) {
            return;
          }
          if (output_dim_index < 0) {
            output_dim_index += this->input_axis_size_;
          }
        }
        output_offset += static_cast<int>(this->output_stride_[i]) * output_dim_index;
      }
      reduction_func(output + output_offset, src);
    }
  };
  ParallelLaunchAutoSearch(task, indices_total_num_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename U, typename V>
bool ScatterValueCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                            const std::vector<kernel::KernelTensor *> &,
                                            const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input = reinterpret_cast<U *>(inputs[kIndex0]->device_ptr());
  auto *indices = reinterpret_cast<V *>(inputs[kIndex2]->device_ptr());
  auto src = static_cast<U>(inputs[kIndex3]->GetValueWithCheck<float>());
  auto *output = reinterpret_cast<U *>(outputs[kIndex0]->device_ptr());
  auto buffer_size = outputs[kIndex0]->size();
  auto memcpy_task = [&](size_t start, size_t end) {
    size_t size = (end - start) * sizeof(U);
    auto ret = memcpy_s(output + start, size, input + start, size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memcpy_s function run error. Error no: " << ret;
    }
  };
  ParallelLaunchAutoSearch(memcpy_task, buffer_size / sizeof(U), this, &parallel_search_info_);

  switch (reduction_type_) {
    case Reduce::REDUCE_NONE:
      return Scatter(ReductionAssignment<U>(), output, indices, src);
    case Reduce::ADD:
      return Scatter(ReductionAdd<U>(), output, indices, src);
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduce_type: " << reduction_type_ << " not support now.";
      return false;
  }
}

#define SCATTER_VALUE_CPU_REG(MS_T, MS_S, U, V)          \
  KernelAttr()                                           \
    .AddInputAttr(MS_T)                                  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)   \
    .AddInputAttr(MS_S)                                  \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32) \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)   \
    .AddOutputAttr(MS_T),                                \
    &ScatterValueCpuKernelMod::LaunchKernel<U, V>

const std::vector<std::pair<KernelAttr, ScatterValueCpuKernelMod::KernelRunFunc>>
  &ScatterValueCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, ScatterValueCpuKernelMod::KernelRunFunc>> func_list = {
    {SCATTER_VALUE_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int32_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int32_t)},

    {SCATTER_VALUE_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
    {SCATTER_VALUE_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)}};
  return func_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, ScatterValue,
                                 []() { return std::make_shared<ScatterValueCpuKernelMod>(kScatterValue); });
}  // namespace scatter_value_cpu
}  // namespace mindspore::kernel
