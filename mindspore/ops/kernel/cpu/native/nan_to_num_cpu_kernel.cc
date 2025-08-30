/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/nan_to_num_cpu_kernel.h"
#include "base/float16.h"

using std::isinf;
using std::isnan;

namespace mindspore {
namespace kernel {
namespace nan_to_num_cpu {
namespace {
constexpr size_t kNanToNumInputsNum = 4;
constexpr size_t kNanToNumOutputsNum = 1;
constexpr auto kNanValueIdx = 1;
constexpr auto kPosinfValueIdx = 2;
constexpr auto kNeginfValueIdx = 3;
}  // namespace

void NanToNumCpuKernelMod::GetInfValues(TypeId input_type, const std::optional<float> &posinf,
                                        const std::optional<float> &neginf, bool posinf_has_value,
                                        bool neginf_has_value) {
  const float FLOAT32_MAX_VALUE = 3.4028235e+38;
  const float FLOAT32_MIN_VALUE = -3.4028235e+38;
  const float FLOAT16_MAX_VALUE = 65504.0;
  const float FLOAT16_MIN_VALUE = -65504.0;
  switch (input_type) {
    case kNumberTypeFloat16:
      posinf_value_ = posinf_has_value ? posinf.value() : FLOAT16_MAX_VALUE;
      neginf_value_ = neginf_has_value ? neginf.value() : FLOAT16_MIN_VALUE;
      break;
    default:
      posinf_value_ = posinf_has_value ? posinf.value() : FLOAT32_MAX_VALUE;
      neginf_value_ = neginf_has_value ? neginf.value() : FLOAT32_MIN_VALUE;
      break;
  }
}

bool NanToNumCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

int NanToNumCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(inputs, outputs)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  const float DEFAULT_NAN = 0.0;

  auto nan_opt = inputs[kNanValueIdx]->GetOptionalValueWithCheck<float>();
  nan_value_ = nan_opt.has_value() ? nan_opt.value() : DEFAULT_NAN;

  auto posinf_opt = inputs[kPosinfValueIdx]->GetOptionalValueWithCheck<float>();
  auto neginf_opt = inputs[kNeginfValueIdx]->GetOptionalValueWithCheck<float>();

  bool posinf_has_value = posinf_opt.has_value();
  bool neginf_has_value = neginf_opt.has_value();
  if (posinf_has_value && neginf_has_value) {
    posinf_value_ = posinf_opt.value();
    neginf_value_ = neginf_opt.value();
  } else {
    auto input_type = inputs[kIndex0]->dtype_id();
    GetInfValues(input_type, posinf_opt, neginf_opt, posinf_has_value, neginf_has_value);
  }
  return 0;
}

template <typename T>
bool NanToNumCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                        const std::vector<kernel::KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kNanToNumInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kNanToNumOutputsNum, kernel_name_);
  auto input = static_cast<T *>(inputs[kIndex0]->device_ptr());
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);
  auto output = static_cast<T *>(outputs[kIndex0]->device_ptr());
  MS_ERROR_IF_NULL_W_RET_VAL(input, false);

  T posinf_value = static_cast<T>(posinf_value_);
  T neginf_value = static_cast<T>(neginf_value_);
  T nan_value = static_cast<T>(nan_value_);
  size_t total = inputs[kIndex0]->size() / sizeof(T);
  auto task = [&input, &output, &posinf_value, &neginf_value, &nan_value](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if (input[i] > static_cast<T>(0) && isinf(input[i])) {
        output[i] = posinf_value;
      } else if (input[i] < static_cast<T>(0) && isinf(input[i])) {
        output[i] = neginf_value;
      } else if (isnan(input[i])) {
        output[i] = nan_value;
      } else {
        output[i] = input[i];
      }
    }
  };
  ParallelLaunchAutoSearch(task, total, this, &parallel_search_info_);
  return true;
}

const std::vector<std::pair<KernelAttr, NanToNumCpuKernelMod::KernelRunFunc>> &NanToNumCpuKernelMod::GetFuncList()
  const {
  static const std::vector<std::pair<KernelAttr, NanToNumCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &NanToNumCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat16),
     &NanToNumCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat16),
     &NanToNumCpuKernelMod::LaunchKernel<float16>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat32),
     &NanToNumCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &NanToNumCpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOptionalInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat32),
     &NanToNumCpuKernelMod::LaunchKernel<float>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NanToNum, NanToNumCpuKernelMod);
}  // namespace nan_to_num_cpu
}  // namespace kernel
}  // namespace mindspore
