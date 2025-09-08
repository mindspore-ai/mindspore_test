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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_H_

#include <string>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"

namespace mindspore {
class CustomKernelData {
 public:
  CustomKernelData() = default;
  virtual ~CustomKernelData() = default;
};

// KernelInputInfo is an interface class.
// There is also a copy of the same code in the ms_op_plugin repository.
// Both sides should be consistent and neither side's code should be modified separately.
class KernelInputInfo {
 public:
  KernelInputInfo() = default;
  virtual ~KernelInputInfo() = default;
  virtual bool IsScalarInput(size_t idx) = 0;

  template <typename T>
  inline T GetKernelInput(size_t) const {
    return T();
  }

  void SetWorkSpace(const std::vector<size_t> &workspace) { workspace_ = workspace; }
  const std::vector<size_t> &WorkSpace() const { return workspace_; }

  void SetKernelData(CustomKernelData *kernel_data) { kernel_data_ = kernel_data; }
  const CustomKernelData *KernelData() const { return kernel_data_; }

  void DestructKernelData() {
    delete kernel_data_;
    kernel_data_ = nullptr;
  }
  virtual size_t GetInputSize() = 0;

  virtual bool GetBoolInput(size_t idx) = 0;
  virtual int64_t GetIntInput(size_t idx) = 0;
  virtual float GetFloatInput(size_t idx) = 0;
  virtual std::string GetStrInput(size_t idx) = 0;

  virtual std::vector<int64_t> GetIntVecInput(size_t idx) = 0;
  virtual std::vector<float> GetFloatVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<int64_t>> GetInt2DVecInput(size_t idx) = 0;
  virtual std::vector<std::vector<float>> GetFloat2DVecInput(size_t idx) = 0;
  virtual int GetInputTypeId(size_t idx) = 0;
  std::vector<size_t> workspace_;

 private:
  CustomKernelData *kernel_data_{nullptr};
};

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
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_CUSTOM_CUSTOM_KERNEL_INPUT_INFO_H_
