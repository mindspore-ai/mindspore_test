/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "kernel/cpu/cross_cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "mindspore/ops/infer/ops_func_impl/cross.h"

namespace mindspore {
namespace kernel {
namespace cross_cpu {
namespace {
const size_t kDataSizeThreshold = 4 * 1024;
const size_t kNumber0 = 0;
const size_t kNumber1 = 1;
const size_t kNumber3 = 3;
}  // namespace

bool CrossCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  constexpr size_t input_num = 3;
  constexpr size_t output_num = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), input_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), output_num, kernel_name_);
  return true;
}

int CrossCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input1_shape_ = inputs[kIndex0]->GetDeviceShapeVector();
  input2_shape_ = inputs[kIndex1]->GetDeviceShapeVector();
  if (input1_shape_.empty() || input2_shape_.empty()) {
    MS_EXCEPTION(ValueError) << "For cross, each input must have at least one dimension, but got input_1 with dim "
                             << input1_shape_.size() << ", input_2 with dim " << input2_shape_.size();
  }
  output_shape_ = outputs[kIndex0]->GetDeviceShapeVector();
  input1_dtype_ = inputs[kIndex0]->dtype_id();
  dim_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  int64_t default_dim = -65530;
  if (dim_ == default_dim) {
    int64_t dim_size_value = 3;
    for (size_t i = 0; i < input1_shape_.size(); i++) {
      if (input1_shape_[i] == dim_size_value) {
        dim_ = static_cast<int64_t>(i);
        break;
      }
      if (i == input1_shape_.size() - 1 && input1_shape_[i] != dim_size_value) {
        MS_EXCEPTION(ValueError) << "The size of inputs dim must be 3,but got" << input1_shape_[i];
      }
    }
  }
  if (dim_ < -static_cast<int64_t>(input1_shape_.size()) || dim_ > static_cast<int64_t>(input1_shape_.size()) - 1) {
    MS_EXCEPTION(ValueError) << "For Cross, dim must be between " << -static_cast<int64_t>(input1_shape_.size())
                             << " and " << static_cast<int64_t>(input1_shape_.size()) - 1 << " , but got " << dim_
                             << ".";
  }
  if (dim_ < 0) {
    dim_ = static_cast<int64_t>(input1_shape_.size()) + dim_;
  }
  return KRET_OK;
}

bool CrossCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  switch (input1_dtype_) {
    case kNumberTypeInt8:
      return LaunchKernel<int8_t>(inputs, outputs);
    case kNumberTypeInt16:
      return LaunchKernel<int16_t>(inputs, outputs);
    case kNumberTypeInt32:
      return LaunchKernel<int32_t>(inputs, outputs);
    case kNumberTypeInt64:
      return LaunchKernel<int64_t>(inputs, outputs);
    case kNumberTypeUInt8:
      return LaunchKernel<uint8_t>(inputs, outputs);
    case kNumberTypeUInt16:
      return LaunchKernel<uint16_t>(inputs, outputs);
    case kNumberTypeUInt32:
      return LaunchKernel<uint32_t>(inputs, outputs);
    case kNumberTypeUInt64:
      return LaunchKernel<uint64_t>(inputs, outputs);
    case kNumberTypeFloat16:
      return LaunchKernel<float16>(inputs, outputs);
    case kNumberTypeFloat32:
      return LaunchKernel<float>(inputs, outputs);
    case kNumberTypeFloat64:
      return LaunchKernel<double>(inputs, outputs);
    case kNumberTypeComplex64:
      return LaunchKernel<std::complex<float>>(inputs, outputs);
    case kNumberTypeComplex128:
      return LaunchKernel<std::complex<double>>(inputs, outputs);
    default:
      MS_EXCEPTION(TypeError) << "Unsupported input data type: " << input1_dtype_;
  }
}

template <typename T>
bool CrossCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  auto input1_data_addr = reinterpret_cast<T *>(inputs[0]->device_ptr());
  int64_t tmp = 1;
  for (size_t i = 0; i < input1_shape_.size(); i++) {
    tmp = tmp * input1_shape_[i];
  }
  size_t input1_data_num = LongToSize(tmp);
  auto input2_data_addr = reinterpret_cast<T *>(inputs[1]->device_ptr());
  auto output_data_addr = reinterpret_cast<T *>(outputs[0]->device_ptr());
  size_t total = input1_data_num / kNumber3;
  const size_t n = input1_shape_.size();
  const int64_t abr_number = kDim3;
  std::vector<std::vector<size_t>> strides(abr_number);
  for (int64_t j = 0; j < abr_number; j++) {
    strides[j].resize(n);
    int64_t stride_tmp = 1;
    const auto &shape = (j == 0) ? input1_shape_ : (j == 1) ? input2_shape_ : output_shape_;
    for (int64_t i = static_cast<int64_t>(n - 1); i >= 0; i--) {
      strides[j][LongToSize(i)] = LongToSize(stride_tmp);
      stride_tmp *= shape[LongToSize(i)];
    }
  }
  size_t input1_data_stride = strides[kDim0][LongToSize(dim_)];
  size_t input2_data_stride = strides[kDim1][LongToSize(dim_)];
  size_t output_data_stride = strides[kDim2][LongToSize(dim_)];
  const size_t pos = 2;
  auto cross_shard = [this, &strides, &output_data_addr, &input1_data_addr, &input2_data_addr, &output_data_stride,
                      &input1_data_stride, &input2_data_stride, &pos](size_t start, size_t end) {
    const size_t input1_data_dim = input1_shape_.size();
    std::vector<int64_t> position_in_dims(input1_data_dim);
    int64_t index_in_curr_dim = SizeToLong(start);
    int64_t input1_data_start = 0;
    int64_t input2_data_start = 0;
    int64_t output_data_start = 0;
    for (int64_t i = 0; i < static_cast<int64_t>(input1_data_dim); i++) {
      if (i == static_cast<int64_t>(dim_)) {
        continue;
      }
      if (input1_shape_[LongToSize(i)] != 0) {
        position_in_dims[LongToSize(i)] = index_in_curr_dim % input1_shape_[LongToSize(i)];
        input1_data_start +=
          (index_in_curr_dim % input1_shape_[LongToSize(i)]) * SizeToLong(strides[kDim0][LongToSize(i)]);
      }
      if (input2_shape_[LongToSize(i)] != 0) {
        input2_data_start +=
          (index_in_curr_dim % input2_shape_[LongToSize(i)]) * SizeToLong(strides[kDim1][LongToSize(i)]);
      }
      if (output_shape_[LongToSize(i)] != 0) {
        output_data_start +=
          (index_in_curr_dim % output_shape_[LongToSize(i)]) * SizeToLong(strides[kDim2][LongToSize(i)]);
      }
      if (input1_shape_[LongToSize(i)] != 0) {
        index_in_curr_dim = index_in_curr_dim / input1_shape_[LongToSize(i)];
      }
    }
    while (start < end) {
      output_data_addr[output_data_start + SizeToLong(kNumber0 * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(kNumber1 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(pos * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(pos * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber1 * input2_data_stride)]);
      output_data_addr[output_data_start + SizeToLong(kNumber1 * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(pos * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber0 * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(kNumber0 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(pos * input2_data_stride)]);
      output_data_addr[output_data_start + SizeToLong(pos * output_data_stride)] =
        static_cast<T>(input1_data_addr[input1_data_start + SizeToLong(kNumber0 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber1 * input2_data_stride)] -
                       input1_data_addr[input1_data_start + SizeToLong(kNumber1 * input1_data_stride)] *
                         input2_data_addr[input2_data_start + SizeToLong(kNumber0 * input2_data_stride)]);
      start++;
      for (size_t i = 0; i < input1_data_dim; i++) {
        if (i == static_cast<size_t>(dim_)) {
          continue;
        }
        position_in_dims[i]++;
        input1_data_start += SizeToLong(strides[kDim0][i]);
        input2_data_start += SizeToLong(strides[kDim1][i]);
        output_data_start += SizeToLong(strides[kDim2][i]);
        if (position_in_dims[i] == input1_shape_[i] && i != (input1_shape_.size() - 1)) {
          input1_data_start -= position_in_dims[i] * SizeToLong(strides[kDim0][i]);
          input2_data_start -= position_in_dims[i] * SizeToLong(strides[kDim1][i]);
          output_data_start -= position_in_dims[i] * SizeToLong(strides[kDim2][i]);
          position_in_dims[i] = 0;
        } else {
          break;
        }
      }
    }
  };
  if (total * sizeof(T) < kDataSizeThreshold) {
    cross_shard(0, total);
  } else {
    CPUKernelUtils::ParallelFor(cross_shard, total);
  }
  return true;
}

std::vector<KernelAttr> CrossCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeUInt8)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt8),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeUInt16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddInputAttr(kNumberTypeUInt32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddInputAttr(kNumberTypeUInt64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeUInt64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeFloat16)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat16),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeFloat32)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat32),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeFloat64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeFloat64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeComplex64)
                                                   .AddInputAttr(kNumberTypeComplex64)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeComplex64),
                                                 KernelAttr()
                                                   .AddInputAttr(kNumberTypeComplex128)
                                                   .AddInputAttr(kNumberTypeComplex128)
                                                   .AddInputAttr(kNumberTypeInt64)
                                                   .AddOutputAttr(kNumberTypeComplex128)};

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Cross, CrossCpuKernelMod);
}  // namespace cross_cpu
}  // namespace kernel
}  // namespace mindspore
