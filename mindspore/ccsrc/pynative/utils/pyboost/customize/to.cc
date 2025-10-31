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

#include "pynative/utils/pyboost/customize/to.h"

#include <utility>
#include <map>
#include "mindapi/base/types.h"
#include "pynative/utils/pyboost/functions/auto_generate/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
device::DeviceType GetDeviceType(Device device) {
  static std::map<Device, device::DeviceType> device_type_map = {{DEVICE_ASCEND, device::DeviceType::kAscend},
                                                                 {DEVICE_NPU_LOWER, device::DeviceType::kAscend},
                                                                 {DEVICE_CPU, device::DeviceType::kCPU},
                                                                 {DEVICE_CPU_LOWER, device::DeviceType::kCPU}};
  auto iter = device_type_map.find(device);
  if (iter == device_type_map.end()) {
    MS_LOG(EXCEPTION) << "Not support to device for " << device;
  }
  return iter->second;
}

Int64ImmPtr GetDeviceByDeviceType(device::DeviceType device_type) {
  static std::map<device::DeviceType, Int64ImmPtr> device_map = {
    {device::DeviceType::kAscend, MakeValue<int64_t>(static_cast<int64_t>(Device::DEVICE_ASCEND))->cast<Int64ImmPtr>()},
    {device::DeviceType::kCPU, MakeValue<int64_t>(static_cast<int64_t>(Device::DEVICE_CPU))->cast<Int64ImmPtr>()}};
  auto iter = device_map.find(device_type);
  if (iter == device_map.end()) {
    MS_LOG(EXCEPTION) << "Not support to device for " << device_type;
  }
  return iter->second;
}

class ToProcessor {
 public:
  ToProcessor(TensorPtr tensor, device::DeviceType device_type, TypeId dtype, bool non_blocking, bool copy)
      : tensor_(std::move(tensor)), device_type_(device_type), dtype_(dtype), non_blocking_(non_blocking), copy_(copy) {
    origin_ = tensor_.get();
  }

  ToProcessor &Contiguous() {
    if (!tensor_->is_contiguous()) {
      tensor_ = pyboost::contiguous(tensor_);
    }
    return *this;
  }

  ToProcessor &Cast() {
    if (tensor_->data_type() != dtype_) {
      auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
      tensor_ = cast(tensor_, dtype_ptr);
    }
    return *this;
  }

  ToProcessor &Device() {
    if (tensor_->device_address()->GetDeviceType() != device_type_) {
      auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
      auto new_output =
        pyboost::empty_like(tensor_, dtype_ptr, GetDeviceByDeviceType(device_type_), std::make_shared<BoolImm>(false));
      auto non_blocking_ptr = MakeValue<bool>(static_cast<int64_t>(non_blocking_))->cast<BoolImmPtr>();
      pyboost::inplace_copy(new_output, tensor_, non_blocking_ptr);
      tensor_ = new_output;
    }
    return *this;
  }

  ToProcessor &Copy() {
    if (origin_ == tensor_.get() && copy_) {
      auto dtype_ptr = MakeValue<int64_t>(static_cast<int64_t>(dtype_))->cast<Int64ImmPtr>();
      auto new_output =
        pyboost::empty_like(tensor_, dtype_ptr, GetDeviceByDeviceType(device_type_), std::make_shared<BoolImm>(false));
      auto non_blocking_ptr = MakeValue<bool>(static_cast<int64_t>(non_blocking_))->cast<BoolImmPtr>();
      pyboost::inplace_copy(new_output, tensor_, non_blocking_ptr);
      tensor_ = new_output;
    }
    return *this;
  }

  const TensorPtr &Get() { return tensor_; }

 private:
  tensor::TensorPtr tensor_;
  device::DeviceType device_type_;
  TypeId dtype_;
  bool non_blocking_;
  bool copy_;
  void *origin_;
};
}  // namespace
tensor::TensorPtr ToDeviceCustomize(const std::shared_ptr<OpRunner> &op, const mindspore::tensor::TensorPtr &self,
                                    const std::optional<mindspore::Int64ImmPtr> &device,
                                    const std::optional<mindspore::Int64ImmPtr> &dtype,
                                    const mindspore::BoolImmPtr &non_blocking, const mindspore::BoolImmPtr &copy) {
  device::DeviceType device_value = device.has_value()
                                      ? GetDeviceType(static_cast<Device>(GetValue<int64_t>(device.value())))
                                      : self->device_address()->GetDeviceType();
  TypeId dtype_value = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : self->data_type();
  bool non_blocking_value = GetValue<bool>(non_blocking);
  bool copy_value = GetValue<bool>(copy);

  MS_LOG(DEBUG) << "Start ToDevice, input is " << self->ToString()
                << " device:" << device::GetDeviceNameByType(device_value)
                << " dtype:" << TypeIdToType(dtype_value)->ToString() << " non_blocking:" << non_blocking_value
                << " copy:" << copy_value;

  auto output = ToProcessor(self, device_value, dtype_value, non_blocking_value, copy_value)
                  .Contiguous()
                  .Cast()
                  .Device()
                  .Copy()
                  .Get();
  op->set_outputs({output});
  MS_LOG(DEBUG) << "End ToDevice";
  return output;
}

tensor::TensorPtr ToDtypeCustomize(const std::shared_ptr<OpRunner> &op, const mindspore::tensor::TensorPtr &self,
                                   const std::optional<mindspore::Int64ImmPtr> &dtype,
                                   const mindspore::BoolImmPtr &non_blocking, const mindspore::BoolImmPtr &copy) {
  device::DeviceType device_value = self->device_address()->GetDeviceType();
  TypeId dtype_value = dtype.has_value() ? static_cast<TypeId>(GetValue<int64_t>(dtype.value())) : self->data_type();
  bool non_blocking_value = GetValue<bool>(non_blocking);
  bool copy_value = GetValue<bool>(copy);

  MS_LOG(DEBUG) << "Start ToDtype, input is " << self->ToString() << " dtype:" << TypeIdToType(dtype_value)->ToString()
                << " non_blocking:" << non_blocking_value << " copy:" << copy_value;
  auto output =
    ToProcessor(self, device_value, dtype_value, non_blocking_value, copy_value).Contiguous().Cast().Copy().Get();
  op->set_outputs({output});
  MS_LOG(DEBUG) << "End ToDevice";
  return output;
}

tensor::TensorPtr ToOtherCustomize(const std::shared_ptr<OpRunner> &op, const mindspore::tensor::TensorPtr &self,
                                   const mindspore::tensor::TensorPtr &other, const mindspore::BoolImmPtr &non_blocking,
                                   const mindspore::BoolImmPtr &copy) {
  device::DeviceType device_value = other->device_address()->GetDeviceType();
  TypeId dtype_value = other->data_type();
  bool non_blocking_value = GetValue<bool>(non_blocking);
  bool copy_value = GetValue<bool>(copy);

  MS_LOG(DEBUG) << "Start ToOther, input is " << self->ToString()
                << " device:" << device::GetDeviceNameByType(device_value)
                << " dtype:" << TypeIdToType(dtype_value)->ToString() << " non_blocking:" << non_blocking_value
                << " copy:" << copy_value;

  auto output = ToProcessor(self, device_value, dtype_value, non_blocking_value, copy_value)
                  .Contiguous()
                  .Cast()
                  .Device()
                  .Copy()
                  .Get();
  op->set_outputs({output});
  MS_LOG(DEBUG) << "End ToDevice";
  return output;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
