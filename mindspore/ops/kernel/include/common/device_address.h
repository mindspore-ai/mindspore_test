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

#ifndef MINDSPORE_DEVICE_TENSOR_H
#define MINDSPORE_DEVICE_TENSOR_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <utility>
#include <mutex>
#include "ir/tensor.h"
#include "ir/dtype.h"
#include "ir/device_sync.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "common/device_type.h"
#include "common/kernel.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan;
class CPUMemoryManager;
class CPUKernelRuntime;
class CPUDeviceContext;
}  // namespace cpu
namespace ascend {
class AscendKernelRuntime;
class AscendRuntimeCore;
class AscendMemoryManager;
class AscendDeviceContext;
class DataDumper;
namespace tasksink {
class TaskGenerator;
}  // namespace tasksink
}  // namespace ascend
namespace gpu {
class GPUKernelRuntime;
class GPUMemoryManager;
class GPUDeviceContext;
}  // namespace gpu
}  // namespace device
class SingleOpInferSession;
class RuntimeUtils;
}  // namespace mindspore

namespace mindspore {
namespace device {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
using kernel::AddressCommon;
using kernel::AddressCommonPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;
using TensorPtr = std::shared_ptr<tensor::Tensor>;
struct StorageInfo {
  void *host_ptr_{nullptr};
  std::string file_name_{""};
  bool host_ptr_mutable_{true};
  bool file_name_mutable_{true};
};

enum class StorageType { kDevice, kHost, kFile };

enum class DeviceAddressStatus {
  kInDevice,
  kInHost,
  kInFile,
  kInDeviceToHost,
  kInHostToDevice,
  kInHostToFile,
  kInFileToHost
};

// The flag of device address.
constexpr size_t kDeviceAddressFlagInit = 0;
// Indicates that it is the device address of ref node.
constexpr size_t kDeviceAddressFlagRefNode = 1;
// Indicates that it is the device address of node which has no user.
constexpr size_t kDeviceAddressFlagNotUsed = 2;
// Indicates that it is the device address of node has init arg and do not need device address.
constexpr size_t kDeviceAddressFlagIgnoreDevicePtr = 4;
// Indicates that it is the ptr of device address is nullptr.
constexpr size_t kDeviceAddressFlagNullptr = 8;

class OPS_KERNEL_COMMON_API DeviceAddress : public mindspore::DeviceSync {
 public:
  using ContinuousDeviceAddressesPtr = std::shared_ptr<std::vector<std::weak_ptr<DeviceAddress>>>;
  explicit DeviceAddress(const KernelTensorPtr &kernel_tensor);

  explicit DeviceAddress(void *ptr, size_t size);
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id);
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index);

  explicit DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id);
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id, const std::string &device_name,
                         uint32_t device_id);
  explicit DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId type_id,
                         const std::string &device_name, uint32_t device_id, uint32_t stream_id);
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id);

  virtual ~DeviceAddress();

  virtual std::string PrintInfo() const;

  virtual bool CopyDeviceToHostWithoutSyncStream(void *dst, size_t dst_size, const void *src, size_t src_size) {
    return true;
  }
  virtual bool AsyncHostToDevice(size_t size, TypeId /* type */, const void *host_ptr) const { return true; }
  virtual bool AsyncHostToDevice(size_t size, TypeId type, const tensor::TensorDataPtr &tensor_data,
                                 const std::string &format) const {
    return true;
  }

  virtual bool AsyncHostToDevice(size_t size, const void *host_ptr) const { return true; }
  virtual bool AsyncDeviceToHost(size_t size, void *host_ptr) const { return true; }

  // Asynchronously copy host memory to device side.
  virtual bool AsyncHostToDevice(const ShapeVector &, size_t, TypeId, const void *, size_t) const { return true; }
  // Asynchronously copy device memory to host side.
  virtual bool AsyncDeviceToHost(const ShapeVector &, size_t, TypeId, void *, size_t) const { return true; }
  // Synchronously copy device memory to device side.
  virtual bool SyncDeviceToDevice(const DeviceSync *) const { return true; }
  virtual bool SyncDeviceToDevice(const ShapeVector &, size_t, TypeId, const void *, const std::string &) const {
    return true;
  }
  // Asynchronously copy device memory to device side.
  virtual bool AsyncDeviceToDevice(const DeviceAddress *) const { return true; }
  virtual bool CopyDeviceToHost(void *dst, const void *src, const size_t &size) const { return true; }
  virtual bool CopyHostToDevice(void *dst, const void *src, const size_t &size) const { return true; }
  virtual void DeviceSynchronizerInit() { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Get kernel tensor pointer.
  const KernelTensorPtr &kernel_tensor() const;
  void set_kernel_tensor(const KernelTensorPtr &kernel_tensor);

  void set_device_synchronizer(const DeviceSynchronizerPtr &device_synchronizer);

  const void *GetPtr() const;
  void set_ptr(void *ptr);
  size_t GetSize() const;
  void SetSize(size_t size);

  std::string format() const;
  void set_format(const std::string &format);
  const std::string &padding_type() const;
  void set_padding_type(const std::string &padding_type);
  TypeId type_id() const;
  void set_type_id(TypeId type_id);
  bool from_mem_pool() const;
  void set_from_mem_pool(bool from_mem_pool) const;
  virtual void set_communication_ptr(uint8_t *communication_ptr);
  bool is_ptr_persisted() const;
  void set_is_ptr_persisted(bool is_ptr_persisted);
  void set_host_shape(const ShapeVector &shape);
  const ShapeVector &host_shape() const;
  void set_device_shape(const ShapeVector &shape);
  const ShapeVector &device_shape() const;
  bool from_persistent_mem() const;
  void set_from_persistent_mem(bool from_persistent_mem);
  bool need_recycle() const;
  void set_need_recycle(bool need_recycle);
  void set_status(DeviceAddressStatus status);
  DeviceAddressStatus status() const;
  virtual DeviceType GetDeviceType() const;
  void *GetMutablePtr() const override;
  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const;

  const TensorStorageInfoPtr GetTensorStorageInfo() const override;
  void set_tensor_storage_info(const TensorStorageInfoPtr &tensor_storage_info);

  const std::string &device_name() const;
  uint32_t device_id() const;

  void set_stream_id(uint32_t stream_id);
  const uint32_t stream_id() const;

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node);
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const;
  void ClearHeldByNodes();

  virtual void SetNodeIndex(const AnfNodePtr &node, size_t out_index);
  KernelWithIndex GetNodeIndex() const;
  size_t IncreaseCounter();
  size_t DecreaseCounter();

  void IncreaseNewRefCount(const std::string &op_name, size_t i = 1);
  size_t DecreaseNewRefCount(const std::string &op_name);
  void set_new_ref_count(size_t new_ref_count) const;
  size_t new_ref_count() const;

  // The related interface of reference count operation.
  void set_original_ref_count(size_t original_ref_count) const override;
  size_t original_ref_count() const override;
  void set_ref_count(size_t ref_count) const override;
  size_t ref_count() const override;
  void ResetRefCount() override;

  void IncreaseOriginalRefCount();
  void DecreaseOriginalRefCount();

  void IncreaseRefCount(size_t increase_cnt);
  size_t DecreaseRefCount();

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_count);

  int32_t dynamic_ref_count() const;

  void IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt);
  void IncreaseDynamicRefCount(const std::string &op_object);
  int32_t DecreaseDynamicRefCount(const std::string &op_object);

  virtual mindspore::tensor::TensorPtr LoadMemToHost(const std::string &tensor_name, const ShapeVector &host_shape,
                                                     TypeId host_type, bool trans_flag, bool async_copy = true) const {
    return nullptr;
  }

  // Return whether DeviceAddress has a valid ptr.
  virtual bool IsPtrValid() const;
  bool IsNotNeedAlloc() const;

  using SyncUserDataHandler = void (*)(DeviceAddress *const device_address);
  // Return the valid device ptr.
  virtual void *GetValidPtr(size_t);

  inline void TouchSyncHandler() {
    if (!need_sync_user_data_ || kernel_tensor_->user_data() == nullptr) {
      return;
    }
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
    if (sync_handler == nullptr) {
      MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
      return;
    }
    (*sync_handler)(this);
    need_sync_user_data_ = false;
  }

  // Offload data from device to host and free device memory
  virtual bool Offload(size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Load data from host to device and free host memory
  virtual bool Load(size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Move data to destination hardware and free resource on source hardware
  virtual bool MoveTo(StorageType, bool, size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  virtual bool Wait() const { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Set host ptr data offloaded to
  virtual void SetOffloadPtr(void *) {}

  // Get offloaded host ptr
  virtual void *GetOffloadPtr() const { return nullptr; }

  virtual void SetStorageInfo(const StorageInfo &) {}
  virtual StorageInfo GetStorageInfo() const { return StorageInfo(); }

  virtual void Swap(DeviceAddress *other);

  virtual void set_swappable(bool) {}
  virtual bool swappable() { return false; }

  // Get user data maintained by the DeviceAddress.
  const UserDataPtr &user_data() const override;

  // Set user data to the DeviceAddress.
  void set_user_data(const UserDataPtr &user_data);

  // Free the ptr in user data when the ref count is 0.
  virtual void ClearUserData() {}

  // The interface of flag.
  size_t flag() const;
  void set_flag(size_t flag);
  void UpdateFlag(size_t flag);
  void ClearFlag(size_t flag);
  std::pair<AnfNodeWeakPtr, size_t> node_index() const;
  void set_deleter(const std::function<void(uint8_t *)> &deleter);
  std::function<void(uint8_t *)> deleter() const;

  // For output of pyexecute kernel, the input data is stored in user data and the handler is used to sync data from
  // user data to device ptr.
  bool need_sync_user_data();
  void set_need_sync_user_data(bool need_sync_user_data);

  const PointerRefCountPtr &pointer_ref_count() const;
  void set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt);
  void set_is_view(bool is_view);
  bool is_view() const;
  AddressCommonPtr address_common() const;
  ContinuousDeviceAddressesPtr continuous_device_addresses() const;
  void set_continuous_device_addresses(const ContinuousDeviceAddressesPtr &continuous_device_addresses);

 protected:
  KernelTensorPtr kernel_tensor_{nullptr};
  // address basic info
  AddressCommonPtr address_common_{nullptr};
  size_t size() const { return address_common_->size_; }

  void *GetDevicePtr() const { return address_common_->pointer_ref_count_->ptr(); }
  void SetDevicePtr(void *ptr) const { address_common_->pointer_ref_count_->set_ptr(ptr); }

  void SetTypeId(TypeId type) const { address_common_->dtype_id_ = type; }
  virtual bool AsyncDeviceToDevice(const ShapeVector &, size_t, TypeId, const void *, const std::string &) const {
    return true;
  }

  ShapeVector device_shape_{};
  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  // The DeviceAddress is held by ValueNodes. These ValueNodes are outputs of forward network.
  // We need to release the device memory when the reference count of the device address in bprop graph is 0.
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes_;
  // Thread lock for ptr_.
  mutable std::recursive_mutex ptr_mutex_;

  bool from_persistent_mem_{false};
  bool need_recycle_{false};

  // The padding type corresponds to data format.
  std::string padding_type_;

  // The device address flag.
  size_t flag_{0};

  // Indicating whether the address is the input of view op.
  // If yes, the device address cannot be reused with the host address in CPU.
  bool is_view_{false};

  // The flag identify where data is stored
  mutable DeviceAddressStatus status_{DeviceAddressStatus::kInDevice};
  // Handler for sync data from user data.
  bool need_sync_user_data_{false};
  // The specified deleter to release memory
  std::function<void(uint8_t *)> deleter_;

  ContinuousDeviceAddressesPtr continuous_device_addresses_{nullptr};

  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUMemoryManager;
  friend class mindspore::device::cpu::CPUKernelRuntime;
  friend class mindspore::device::cpu::CPUDeviceContext;
  friend class mindspore::device::gpu::GPUKernelRuntime;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::gpu::GPUDeviceContext;
  friend class mindspore::device::ascend::AscendKernelRuntime;
  friend class mindspore::device::ascend::AscendRuntimeCore;
  friend class mindspore::device::ascend::AscendMemoryManager;
  friend class mindspore::device::ascend::AscendDeviceContext;
  friend class mindspore::device::ascend::DataDumper;
  friend class mindspore::SingleOpInferSession;
  friend class mindspore::RuntimeUtils;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
