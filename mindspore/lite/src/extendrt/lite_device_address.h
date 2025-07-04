/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_LITE_DEVICE_ADDRESS_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_LITE_DEVICE_ADDRESS_H_

#include <memory>
#include <string>
#include <utility>

#include "abstract/abstract_function.h"
#include "common/device_address.h"
#include "common/kernel_tensor.h"
#include "common/kernel_utils.h"
#include "common/common_utils.h"
#include "kernel/framework_utils.h"

namespace mindspore {
namespace runtime {
namespace test {
using abstract::AbstractFuncUnion;
using abstract::AbstractTensor;
using abstract::AbstractTensorPtr;
using abstract::AnalysisContext;
using abstract::FuncGraphAbstractClosure;
using device::DeviceAddress;
using device::DeviceAddressPtr;
using device::DeviceType;
using kernel::AddressPtr;
using kernel::KernelTensorPtr;

class TestDeviceAddress : public DeviceAddress {
 public:
  TestDeviceAddress() : DeviceAddress() {}
  TestDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  TestDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                    uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  ~TestDeviceAddress() {}
  virtual bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr,
                                bool sync_on_demand) const {
    return true;
  }
  virtual bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                                const std::string &format) const {
    return true;
  }
  virtual void ClearDeviceMemory() {}
  DeviceType GetDeviceType() const override { return DeviceType::kCPU; }

  void set_data(tensor::TensorDataPtr &&data) override { data_ = std::move(data); }

  const tensor::TensorDataPtr &data() const override { return data_; }

  bool has_data() const override { return data_ != nullptr; }

 private:
  // the data for numpy object.
  tensor::TensorDataPtr data_;
};
}  // namespace test
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_LITE_DEVICE_ADDRESS_H_
