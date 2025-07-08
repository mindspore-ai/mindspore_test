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

#ifndef MINDSPORE_LITE_SRC_EXTENDRT_LITE_DEVICE_ADDRESS_H_
#define MINDSPORE_LITE_SRC_EXTENDRT_LITE_DEVICE_ADDRESS_H_

#include <memory>
#include <string>
#include <utility>

#include "common/device_address.h"

namespace mindspore {
namespace runtime {
namespace test {
using device::DeviceAddress;
using device::DeviceAddressPtr;
using device::DeviceType;

class TestDeviceAddress : public DeviceAddress {
 public:
  TestDeviceAddress() = delete;
  TestDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                    uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  ~TestDeviceAddress() {}
  void ClearDeviceMemory() {}
  DeviceType GetDeviceType() const { return DeviceType::kCPU; }

  void set_data(tensor::TensorDataPtr &&data) { data_ = std::move(data); }

  const tensor::TensorDataPtr &data() const { return data_; }

  bool has_data() const { return data_ != nullptr; }

 private:
  // the data for numpy object.
  tensor::TensorDataPtr data_;
};
}  // namespace test
}  // namespace runtime
}  // namespace mindspore
#endif  // MINDSPORE_LITE_SRC_EXTENDRT_LITE_DEVICE_ADDRESS_H_
