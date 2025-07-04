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

#include "kernel/cpu/hshrink_cpu_kernel.h"
#include <algorithm>
#include "common/ms_factory.h"
#include "kernel/cpu/nnacl/errorcode.h"
#include "kernel/cpu/nnacl/fp32/activation_fp32.h"

namespace mindspore {
namespace kernel {
namespace hshrink_cpu {
namespace {
constexpr size_t kHShrinkInputsNum = 2;
constexpr size_t kHShrinkOutputsNum = 1;

const std::vector<KernelAttr> kernel_attr = {{KernelAttr()
                                                .AddInputAttr(kNumberTypeFloat32)
                                                .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
                                                .AddOutputAttr(kNumberTypeFloat32)}};
}  // namespace

bool HShrinkCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kHShrinkInputsNum || outputs.size() != kHShrinkOutputsNum) {
    MS_LOG(ERROR) << kernel_name_ << ": input and output size should be " << kHShrinkInputsNum << " and "
                  << kHShrinkOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  auto input_type_id = inputs[0]->dtype_id();

  if (input_type_id != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "HShrink kernel does not support " << TypeIdToString(input_type_id);
    return false;
  }
  unit_size_ = sizeof(float);
  return true;
}

int HShrinkCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  MS_EXCEPTION_IF_CHECK_FAIL(unit_size_ != 0, "For HShrink, the value of [unit_size_] must not be 0!");
  input_elements_ = inputs[0]->size() / unit_size_;
  lambd = inputs[kIndex1]->GetValueWithCheck<float>();
  return KRET_OK;
}

bool HShrinkCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                 const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHShrinkInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHShrinkOutputsNum, kernel_name_);
  auto *input = GetDeviceAddress<float>(inputs, kIndex0);
  auto *output = GetDeviceAddress<float>(outputs, kIndex0);

  auto task = [input, output, this](size_t start, size_t end) {
    auto ret = HardShrink(input + start, SizeToInt(end - start), output + start, lambd);
    if (ret != NNACL_OK) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', call NNACL HShrink function failed. Error code: " << ret;
      return false;
    }
    return true;
  };
  ParallelLaunchAutoSearch(task, input_elements_, this, &parallel_search_info_, pool_);
  return true;
}

std::vector<KernelAttr> HShrinkCpuKernelMod::GetOpSupport() { return kernel_attr; }

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, HShrink, HShrinkCpuKernelMod);
}  // namespace hshrink_cpu
}  // namespace kernel
}  // namespace mindspore
