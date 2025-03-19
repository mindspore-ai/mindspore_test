/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_

#include <string>
#include <vector>
#include "common/device_address.h"
#include "runtime/hardware/device_context.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/device/res_manager/loadable_device_address.h"

using ShapeVecotr = std::vector<int>;

namespace mindspore {
namespace device {
namespace gpu {
class GPUDeviceAddress : public LoadableDeviceAddress {
 public:
  explicit GPUDeviceAddress(const KernelTensorPtr &kernel_tensor) : LoadableDeviceAddress(kernel_tensor) {
    SetDevicePtrDeleter();
  }
  GPUDeviceAddress(void *ptr, size_t size) : LoadableDeviceAddress(ptr, size) { SetDevicePtrDeleter(); }
  GPUDeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : LoadableDeviceAddress(ptr, size, format, type_id) {
    SetDevicePtrDeleter();
  }
  GPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const KernelWithIndex &node_index)
      : LoadableDeviceAddress(ptr, size, format, type_id, node_index) {
    SetDevicePtrDeleter();
  }
  GPUDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id, const std::string &device_name,
                   uint32_t device_id)
      : LoadableDeviceAddress(ptr, size, format, type_id, device_name, device_id) {
    SetDevicePtrDeleter();
  }
  GPUDeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId type_id,
                   const std::string &device_name, uint32_t device_id, uint32_t stream_id)
      : LoadableDeviceAddress(ptr, size, shape_vector, format, type_id, device_name, device_id, stream_id) {
    SetDevicePtrDeleter();
  }
  ~GPUDeviceAddress() override;
  void DeviceSynchronizerInit() override;

  bool SyncDeviceToHost(size_t size, void *host_ptr) const override;
  bool SyncHostToDevice(size_t size, const void *host_ptr) const override;
  bool SyncDeviceToHost(const ShapeVector &shape, size_t size, TypeId type, void *host_ptr) const override;
  bool SyncHostToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *host_ptr,
                        const std::string &format) const override;
  bool SyncDeviceToDevice(const DeviceSync *src_device_addr) const override;
  bool AsyncDeviceToDevice(const DeviceAddress *src_device_addr) const override;
  bool SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                          const std::string &format) const override;
  bool CopyDeviceToHost(void *dst, const void *src, const size_t &size) const override;
  bool CopyHostToDevice(void *dst, const void *src, const size_t &size) const override;

  void ClearDeviceMemory() override;
  DeviceType GetDeviceType() const override { return DeviceType::kGPU; }
  mindspore::tensor::TensorPtr LoadMemToHost(const std::string &tensor_name, const ShapeVector &host_shape,
                                             TypeId host_type, bool trans_flag, bool async_copy = true) const override;

  // Asynchronously copy host memory to device side.
  bool AsyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr,
                         size_t stream_id) const override;
  bool AsyncHostToDevice(size_t size, TypeId type, const tensor::TensorDataPtr &tensor_data,
                         const std::string &format) const override;

  // Asynchronously copy device memory to host side.
  bool AsyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr, size_t stream_id) const override;

  bool AsyncHostToDevice(size_t size, const void *host_ptr) const override;

  bool AsyncDeviceToHost(size_t size, void *host_ptr) const override;

  void ClearUserData() override;

 protected:
  bool CopyDeviceToHost(void *dst, const void *src, size_t size, bool async, size_t stream_id) const override;
  bool CopyHostToDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id) const override;
  bool AsyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                           const std::string &format) const override;

 private:
  HalResBase *GetHalRes() const {
    device::ResKey res_key{device::GetDeviceTypeByName(device_name()), device_id()};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    return res_manager;
  }
  bool CopyBetweenHostDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id,
                             bool host_to_device) const;

  // Set a device pointer destructor to kernel tensor, used to release resource reclaiming of the device pointer
  // automatically when DeviceAddress destructed.
  void SetDevicePtrDeleter();
};
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_GPU_GPU_DEVICE_ADDRESS_H_
