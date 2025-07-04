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

#include "kernel/cpu/is_close_cpu_kernel.h"
#include <functional>
#include <algorithm>
#include <memory>
#include "mindspore/ops/infer/ops_func_impl/isclose.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace is_close_cpu {
namespace {
constexpr size_t kIsCloseInputsNum = 5;
constexpr size_t kIsCloseOutputsNum = 1;
template <typename T>
inline bool IsClose(T a, T b, float rtol, float atol, bool equal_nan) {
  if (std::equal_to<T>()(a, b)) {
    return true;
  }
  if (equal_nan && std::isnan(a) && std::isnan(b)) {
    return true;
  }
  if (std::equal_to<float>()(atol, 0) && std::equal_to<float>()(rtol, 0)) {
    return false;
  }
  auto left_side = std::abs(a - b);
  auto right_side = atol + (rtol * std::abs(b));
  return std::isfinite(left_side) && left_side <= right_side;
}
}  // namespace
bool IsCloseCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsCloseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsCloseOutputsNum, kernel_name_);
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int IsCloseCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  rtol_ = inputs[kIndex2]->GetValueWithCheck<float>();
  atol_ = inputs[kIndex3]->GetValueWithCheck<float>();
  equal_nan_ = inputs[kIndex4]->GetValueWithCheck<bool>();
  auto input_shape = LongVecToSizeVec(inputs.at(kIndex0)->GetShapeVector());
  auto other_shape = LongVecToSizeVec(inputs.at(kIndex1)->GetShapeVector());
  auto output_shape = LongVecToSizeVec(outputs.at(kIndex0)->GetShapeVector());
  has_null_input_ = CheckNullInput(input_shape);
  has_null_input_ = has_null_input_ || CheckNullInput(other_shape);
  if (has_null_input_) {
    return KRET_OK;
  }
  is_need_broadcast_ = input_shape != other_shape;
  if (is_need_broadcast_) {
    GetBroadCastIndex(input_shape, output_shape, &index_list1_);
    GetBroadCastIndex(other_shape, output_shape, &index_list2_);
  }
  return KRET_OK;
}

template <typename T>
bool IsCloseCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                       const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsCloseInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsCloseOutputsNum, kernel_name_);
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto other = GetDeviceAddress<T>(inputs, kIndex1);
  auto output = GetDeviceAddress<bool>(outputs, kIndex0);

  if (has_null_input_) {
    return true;
  }

  CTask task;
  if (!is_need_broadcast_) {
    task = [this, &input, &other, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
          auto a = static_cast<float>(input[i]);
          auto b = static_cast<float>(other[i]);
          output[i] = IsClose<float>(a, b, rtol_, atol_, equal_nan_);
        } else {
          output[i] = IsClose(input[i], other[i], rtol_, atol_, equal_nan_);
        }
      }
    };
  } else {
    task = [this, &input, &other, &output](size_t start, size_t end) {
      for (size_t i = start; i < end; i++) {
        auto idx1 = index_list1_[i];
        auto idx2 = index_list2_[i];
        if constexpr (!std::is_same_v<T, float> && !std::is_same_v<T, double>) {
          auto a = static_cast<float>(input[idx1]);
          auto b = static_cast<float>(other[idx2]);
          output[i] = IsClose<float>(a, b, rtol_, atol_, equal_nan_);
        } else {
          output[i] = IsClose(input[idx1], other[idx2], rtol_, atol_, equal_nan_);
        }
      }
    };
  }
  size_t elem_num = outputs[kIndex0]->size() / sizeof(bool);
  ParallelLaunch(task, elem_num, 0, this, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, IsCloseCpuKernelMod::KernelRunFunc>> IsCloseCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kNumberTypeInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kNumberTypeInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<double>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kNumberTypeUInt8)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kNumberTypeUInt16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kNumberTypeUInt32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kNumberTypeUInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeBool),
   &IsCloseCpuKernelMod::LaunchKernel<uint64_t>},
};

const std::vector<std::pair<KernelAttr, IsCloseCpuKernelMod::KernelRunFunc>> &IsCloseCpuKernelMod::GetFuncList() const {
  return func_list_;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IsClose, IsCloseCpuKernelMod);
}  // namespace is_close_cpu
}  // namespace kernel
}  // namespace mindspore
