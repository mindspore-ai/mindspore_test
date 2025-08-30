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

#include "kernel/cpu/native/scatter_cpu_kernel.h"
#include <algorithm>
#include <limits>
#include <functional>

#include "mindspore/ops/infer/ops_func_impl/tensor_scatter_elements.h"

namespace mindspore::kernel {
namespace scatter_cpu {
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

bool ScatterCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (kernel_name_ == kTensorScatterElements) {
    input_name_ = "data";
    axis_name_ = "axis";
    index_name_ = "indices";
    src_name_ = "updates";
    axis_idx_ = kIndex3;
    index_idx_ = kIndex1;
    src_idx_ = kIndex2;
  } else if (kernel_name_ != kScatter) {
    MS_LOG(ERROR) << "Need to be " << kernel_type_ << " but got kernel name as " << kernel_name_;
    return false;
  }
  reduction_type_ = inputs[reduce_idx_]->GetValueWithCheck<Reduce>();
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int ScatterCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = KernelMod::Resize(inputs, outputs)) != 0) {
    return ret;
  }
  auto input_shape = inputs[input_idx_]->GetShapeVector();
  input_dims_ = input_shape.size();
  if (input_dims_ < 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of '" << input_name_
                  << "' should be greater than or equal to 1, but got " << input_dims_ << ".";
    return KRET_RESIZE_FAILED;
  }
  auto index_shape = inputs[index_idx_]->GetShapeVector();
  auto src_shape = inputs[src_idx_]->GetShapeVector();
  if (index_shape != src_shape) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the shape of '" << index_name_ << "' and the shape of '"
                  << src_name_ << "' should be same, but got " << index_name_ << " shape: " << index_shape << "; "
                  << src_name_ << " shape: " << src_shape << ".";
    return KRET_RESIZE_FAILED;
  }
  if (input_dims_ != index_shape.size()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of '" << input_name_ << "', '" << index_name_
                  << "' should be same, but got " << input_name_ << " dims: " << input_dims_ << "; " << index_name_
                  << " dims: " << index_shape.size() << ". ";
    return KRET_RESIZE_FAILED;
  }

  axis_ = inputs[axis_idx_]->GetValueWithCheck<int64_t>();
  if (axis_ < 0) {
    axis_ += static_cast<int64_t>(input_dims_);
  }

  if (axis_ >= static_cast<int64_t>(input_dims_) || axis_ < 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the '" << axis_name_
                  << "' should be less than input dims and greater than or equal 0, but got " << axis_
                  << ", while input dims is: " << input_dims_;
    return KRET_RESIZE_FAILED;
  }

  for (size_t i = 0; i < input_dims_; ++i) {
    if (axis_ != static_cast<int64_t>(i) && input_shape[i] < index_shape[i]) {
      MS_LOG(ERROR) << "For '" << kernel_name_
                    << "', the indices dims should be less than input dims, but got indices dim is: " << index_shape[i]
                    << " at axis: " << i << ", while input dim is:" << input_shape[i];
      return KRET_RESIZE_FAILED;
    }
  }

  input_axis_size_ = SizeToInt(input_shape[axis_]);
  input_size_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  indices_total_num_ = std::accumulate(index_shape.begin(), index_shape.end(), size_t(1), std::multiplies<size_t>());

  // calculate indices_stride
  indices_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    indices_stride_[i - 1] = indices_stride_[i] * index_shape[i];
  }

  // calculate output_stride
  output_stride_.resize(input_dims_, 1);
  for (size_t i = input_dims_ - 1; i > 0; --i) {
    output_stride_[i - 1] = output_stride_[i] * static_cast<size_t>(input_shape[i]);
  }
  return KRET_OK;
}

template <typename T, typename S, typename ReductionT>
bool ScatterCpuKernelMod::Scatter(const ReductionT &reduction_func, T *output, const S *indices, const T *updates) {
  auto task = [reduction_func, output, indices, updates, this](size_t start, size_t end) {
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
      reduction_func(output + output_offset, *(updates + index));
    }
  };
  ParallelLaunchAutoSearch(task, indices_total_num_, this, &parallel_search_info_, pool_);
  return true;
}

template <typename T, typename S>
bool ScatterCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                       const std::vector<kernel::KernelTensor *> &,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  auto *input = reinterpret_cast<T *>(inputs[input_idx_]->device_ptr());
  auto *indices = reinterpret_cast<S *>(inputs[index_idx_]->device_ptr());
  auto *src = reinterpret_cast<T *>(inputs[src_idx_]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[kIndex0]->device_ptr());
  auto buffer_size = outputs[kIndex0]->size();
  auto memcpy_task = [&](size_t start, size_t end) {
    size_t size = (end - start) * sizeof(T);
    auto ret = memcpy_s(output + start, size, input + start, size);
    if (ret != EOK) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', it's memcpy_s function run error. Error no: " << ret;
    }
  };
  ParallelLaunchAutoSearch(memcpy_task, buffer_size / sizeof(T), this, &parallel_search_info_);

  switch (reduction_type_) {
    case Reduce::REDUCE_NONE:
      return Scatter(ReductionAssignment<T>(), output, indices, src);
    case Reduce::ADD:
      return Scatter(ReductionAdd<T>(), output, indices, src);
    default:
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', reduce_type: " << reduction_type_ << " not support now.";
      return false;
  }
}

#define SCATTER_CPU_REG(MS_T, MS_S, T, S)              \
  KernelAttr()                                         \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddInputAttr(MS_S)                                \
    .AddInputAttr(MS_T)                                \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64) \
    .AddOutputAttr(MS_T),                              \
    &ScatterCpuKernelMod::LaunchKernel<T, S>

#define TENSOR_SCATTER_ELEMENTS_CPU_REG(MS_T, MS_S, T, S) \
  KernelAttr()                                            \
    .AddInputAttr(MS_T)                                   \
    .AddInputAttr(MS_S)                                   \
    .AddInputAttr(MS_T)                                   \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)    \
    .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)    \
    .AddOutputAttr(MS_T),                                 \
    &ScatterCpuKernelMod::LaunchKernel<T, S>

const std::vector<std::pair<KernelAttr, ScatterCpuKernelMod::KernelRunFunc>> &ScatterCpuKernelMod::GetFuncList() const {
  static std::vector<std::pair<KernelAttr, ScatterCpuKernelMod::KernelRunFunc>> func_list{};
  if (kernel_type_ == kScatter) {
    func_list = {{SCATTER_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int32_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int32_t)},

                 {SCATTER_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
                 {SCATTER_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)}};
  } else {
    func_list = {{TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeBool, kNumberTypeInt32, bool, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt8, kNumberTypeInt32, int8_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt32, kNumberTypeInt32, int32_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt32, kNumberTypeInt32, uint32_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat16, kNumberTypeInt32, float16, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat32, kNumberTypeInt32, float, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat64, kNumberTypeInt32, double, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt64, kNumberTypeInt32, int64_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt64, kNumberTypeInt32, uint64_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt16, kNumberTypeInt32, int16_t, int32_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt16, kNumberTypeInt32, uint16_t, int32_t)},

                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeBool, kNumberTypeInt64, bool, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt8, kNumberTypeInt64, int8_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt32, kNumberTypeInt64, int32_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt32, kNumberTypeInt64, uint32_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat16, kNumberTypeInt64, float16, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat32, kNumberTypeInt64, float, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeFloat64, kNumberTypeInt64, double, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt64, kNumberTypeInt64, int64_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt64, kNumberTypeInt64, uint64_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeInt16, kNumberTypeInt64, int16_t, int64_t)},
                 {TENSOR_SCATTER_ELEMENTS_CPU_REG(kNumberTypeUInt16, kNumberTypeInt64, uint16_t, int64_t)}};
  }
  return func_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, TensorScatterElements,
                                 []() { return std::make_shared<ScatterCpuKernelMod>(kTensorScatterElements); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeCpuKernelMod, Scatter,
                                 []() { return std::make_shared<ScatterCpuKernelMod>(kScatter); });
}  // namespace scatter_cpu
}  // namespace mindspore::kernel
