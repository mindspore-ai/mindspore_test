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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_

#include <string>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "kernel/cpu/custom/kernel_mod_impl/custom_kernel_input_info.h"

namespace mindspore {
class KernelInputInfoImpl : public KernelInputInfo {
 public:
  KernelInputInfoImpl() = default;
  virtual ~KernelInputInfoImpl() = default;
  void SetKernelInput(const std::vector<kernel::KernelTensor *> &inputs) { inputs_ = inputs; }
  size_t GetInputSize() { return inputs_.size(); }
  bool IsScalarInput(size_t idx) final { return inputs_[idx]->type_id() != TypeId::kObjectTypeTensorType; }

  bool GetBoolInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<bool>(); }

  int64_t GetIntInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<int64_t>(); }

  float GetFloatInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<float>(); }

  std::string GetStrInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<std::string>(); }

  std::vector<int64_t> GetIntVecInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<std::vector<int64_t>>(); }

  std::vector<float> GetFloatVecInput(size_t idx) { return inputs_[idx]->GetValueWithCheck<std::vector<float>>(); }

  std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) {
    return inputs_[idx]->GetValueWithCheck<std::vector<std::vector<int64_t>>>();
  }

  std::vector<std::vector<float>> GetFloat2DVecInput(size_t idx) {
    return inputs_[idx]->GetValueWithCheck<std::vector<std::vector<float>>>();
  }

  int GetInputTypeId(size_t idx) { return static_cast<int>(inputs_[idx]->dtype_id()); }

 private:
  std::vector<kernel::KernelTensor *> inputs_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_IMPL_H_
