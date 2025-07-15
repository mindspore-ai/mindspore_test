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
#include <optional>
#include "ir/tensor.h"
#include "ir/dtype.h"
#include "ir/device_sync.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "common/device_type.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan;
class CPUMemoryManager;
class CPUDeviceContext;
}  // namespace cpu
namespace ascend {
class AscendRuntimeCore;
class AscendMemoryManager;
class DataDumper;
namespace tasksink {
class TaskGenerator;
}  // namespace tasksink
}  // namespace ascend
namespace gpu {
class GPUMemoryManager;
class GPUDeviceContext;
}  // namespace gpu
}  // namespace device
class SingleOpInferSession;
class RuntimeUtils;
}  // namespace mindspore

namespace mindspore {
// PointerRefCount encapsulates pointer and reference count-related operations, and supports custom deleter to free
// resources. In Ref scenarios, KernelTensor of different DeviceAddress may hold the same PointerRefCount object.
class PointerRefCount {
 public:
  // The arguments are pointer and a bool variable that identifies whether pointer is from the memory pool.
  using Deleter = std::function<void(void *, bool)>;

  PointerRefCount() = default;
  explicit PointerRefCount(void *ptr) : ptr_(ptr) {}
  PointerRefCount(void *ptr, const Deleter &deleter) : ptr_(ptr), deleter_(deleter) {}

  PointerRefCount(const PointerRefCount &) = delete;
  PointerRefCount &operator=(const PointerRefCount &) = delete;

  ~PointerRefCount() {
    try {
      if (ptr_ != nullptr && deleter_) {
        deleter_(ptr_, from_mem_pool_);
      }
      ptr_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "PointerRefCount destructed failed: " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "PointerRefCount destructed failed.";
    }
  }

  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " ptr:" << ptr_ << " from mem pool:" << from_mem_pool_ << " origin ref count:" << original_ref_count_
        << " ref count:" << ref_count_ << " dynamic ref count:" << dynamic_ref_count_
        << " new ref count:" << new_ref_count_;
    return ofs.str();
  }

  // Get raw pointer.
  void *ptr() const { return ptr_; }
  // Set raw pointer.
  void set_ptr(void *ptr) { ptr_ = ptr; }

  // Get whether pointer in PointerRefCount is allocated from the memory pool.
  bool from_mem_pool() const { return from_mem_pool_; }
  // Set whether pointer in PointerRefCount is allocated from the memory pool.
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }

  // The related interface of static reference count operation.
  void set_original_ref_count(size_t original_ref_count) { original_ref_count_ = original_ref_count; }
  size_t original_ref_count() const { return original_ref_count_; }
  void set_ref_count(size_t ref_count) { ref_count_ = ref_count; }
  size_t ref_count() const { return ref_count_.load(); }
  void IncreaseOriginalRefCount() {
    if (original_ref_count_ < SIZE_MAX) {
      original_ref_count_++;
    }
  }
  void DecreaseOriginalRefCount() {
    if ((original_ref_count_ < SIZE_MAX) && (original_ref_count_ > 0)) {
      original_ref_count_--;
    }
  }

  void IncreaseRefCount(size_t increase_cnt) {
    if (ref_count() < SIZE_MAX && (SIZE_MAX - ref_count()) > increase_cnt) {
      ref_count_ += increase_cnt;
      return;
    }
    MS_LOG(EXCEPTION) << "The reference count is:" << ref_count() << ", and can't add: " << increase_cnt << " more.";
  }
  size_t DecreaseRefCount() { return --ref_count_; }
  void ResetRefCount() { ref_count_ = original_ref_count_; }

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_count) { dynamic_ref_count_ = dynamic_ref_count; }
  int32_t dynamic_ref_count() const { return dynamic_ref_count_; }

  void IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt) {
    if (dynamic_ref_count_ < INT32_MAX && (INT32_MAX - dynamic_ref_count_) > increase_cnt) {
      auto ret = dynamic_ref_count_.fetch_add(increase_cnt) + increase_cnt;
      MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << ret << " for ptr:" << ptr();
      return;
    }
    MS_LOG(EXCEPTION) << "The dynamic reference count is:" << dynamic_ref_count_ << ", and can't add: " << increase_cnt
                      << " more.";
  }
  void IncreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ < INT32_MAX) {
      auto ret = ++dynamic_ref_count_;
      MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << ret << " for ptr:" << ptr();
    }
  }
  int32_t DecreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ <= 0) {
      MS_LOG(EXCEPTION) << "The dynamic reference count is invalid value:" << dynamic_ref_count_;
    }
    auto ret = --dynamic_ref_count_;
    MS_LOG(DEBUG) << op_object << " The dynamic ref count decreases to:" << ret << " for ptr:" << ptr();
    return ret;
  }

  // Get pointer resource destructor.
  Deleter deleter() const { return deleter_; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { deleter_ = deleter; }

  bool is_ptr_persisted() const { return is_ptr_persisted_; }
  void set_is_ptr_persisted(bool is_ptr_persisted) { is_ptr_persisted_ = is_ptr_persisted; }

  // New ref count interface.
  void IncreaseNewRefCount(size_t i = 1) {
    if (new_ref_count_ < SIZE_MAX) {
      new_ref_count_ += i;
    }
  }
  size_t DecreaseNewRefCount() {
    if (new_ref_count_ == 0) {
      MS_LOG(EXCEPTION) << "Failed to decrease ref count:" << this;
    }
    if (new_ref_count_ == SIZE_MAX) {
      return SIZE_MAX;
    }
    return --new_ref_count_;
  }
  void set_new_ref_count(size_t new_ref_count) { new_ref_count_ = new_ref_count; }
  size_t new_ref_count() const { return new_ref_count_.load(); }

 private:
  void *ptr_{nullptr};

  // Whether ptr_  is allocated from the memory pool.
  bool from_mem_pool_{false};

  // The static reference count, the value can be calculated at compile phase.
  size_t original_ref_count_{1};
  // The current reference count value, it will be decreased in the running, and reset by original_ref_count_ when it is
  // zero.
  std::atomic<size_t> ref_count_{1};

  std::atomic<size_t> new_ref_count_{0};

  // The dynamic reference count, the value can be calculated at compile phase.
  std::atomic_int32_t dynamic_ref_count_{INT32_MAX};

  // The pointer resource destructor.
  Deleter deleter_;

  // The device address of the node that owns the device address cannot be updated and replaced.
  // Application scenario: set to true when the hardware execution mode requires that ptr cannot be changed during
  // execution.
  bool is_ptr_persisted_{false};
};
using PointerRefCountPtr = std::shared_ptr<PointerRefCount>;

struct AddressCommon {
  AddressCommon() { pointer_ref_count_ = std::make_shared<PointerRefCount>(); }
  AddressCommon(void *device_ptr, size_t size)
      : pointer_ref_count_(std::make_shared<PointerRefCount>(device_ptr)), size_(size) {}
  AddressCommon(void *device_ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId dtype_id,
                const std::string &device_name, uint32_t device_id, uint32_t stream_id = 0)
      : pointer_ref_count_(std::make_shared<PointerRefCount>(device_ptr)),
        stream_id_(stream_id),
        size_(size),
        format_(format),
        dtype_id_(dtype_id),
        device_name_(device_name),
        device_id_(device_id),
        shape_vector_(shape_vector) {}
  AddressCommon(const AddressCommon &other) {
    pointer_ref_count_ =
      other.pointer_ref_count_ != nullptr
        ? std::make_shared<PointerRefCount>(other.pointer_ref_count_->ptr(), other.pointer_ref_count_->deleter())
        : std::make_shared<PointerRefCount>();
    tensor_storage_info_ = other.tensor_storage_info_;
    stream_id_ = other.stream_id_;
    size_ = other.size_;
    format_ = other.format_;
    dtype_id_ = other.dtype_id_;
    device_id_ = other.device_id_;
    device_name_ = other.device_name_;
    dtype_id_ = other.dtype_id_;
    shape_vector_ = other.shape_vector_;
    managed_by_somas_ = other.managed_by_somas_;
  }
  AddressCommon &operator=(const AddressCommon &) = delete;

  std::string ToString() const {
    std::ostringstream ofs;
    ofs << " size:" << size_ << " tensor storage info:" << tensor_storage_info_;
    if (tensor_storage_info_ != nullptr) {
      ofs << tensor_storage_info_->ToString();
    }
    ofs << " size:" << size_ << " format:" << format_ << " dtype:" << dtype_id_ << " device id:" << device_id_
        << " device name:" << device_name_ << " shape vector:{";
    std::for_each(shape_vector_.begin(), shape_vector_.end(), [&ofs](ShapeValueDType axis) { ofs << axis << " "; });
    ofs << "} point ref count:";
    if (pointer_ref_count_ == nullptr) {
      ofs << "0";
    } else {
      ofs << pointer_ref_count_->ToString();
    }
    return ofs.str();
  }

  PointerRefCountPtr pointer_ref_count_;
  TensorStorageInfoPtr tensor_storage_info_{nullptr};
  uint32_t stream_id_{0};
  size_t size_{0};
  Format format_{Format::DEFAULT_FORMAT};
  // The data enum type id of the KernelTensor.
  TypeId dtype_id_{kTypeUnknown};
  // The device target name, such as "GPU","Ascend".
  std::string device_name_;
  // Represents the device card id associated with the KernelTensor.
  uint32_t device_id_{0};
  // The origin flatten shape vector for Tensor/Scalar/Tuple/List.
  // 1. For Tensor type, means its shape. For example, a Tensor with shape (8, 16), shape_vector_ is {8, 16}.
  // 2. For Scalar type, shape_vector_ is an empty ShapeVector, i.e. {}.
  // 3. For Tuple/List (all elements must be Tensor with same shape or Scalar) type, the shape_vector_
  // consists of the element number and the shape of element in Tuple/List. For example, if a Tuple of the structure
  // ((8,16), (8,16)) contains two Tensors of shape (8, 16), then shape_vector_ is {2, 8, 16}, 2 means elements
  // number in Tuple/List. A Tuple with a structure such as ((), ()) that contains two Scalar, the shape_vector_ of
  // this Tuple is {2}.
  ShapeVector shape_vector_{};
  bool managed_by_somas_{false};
};
using AddressCommonPtr = std::shared_ptr<AddressCommon>;

enum class NeedAllocateHeteRes : int64_t { NoNeedHeteRes = 0, NeedHostMem = 1, NeedDiskFile = 2 };
struct HeterogeneousInfo {
  // Address on cpu ddr when the KernelTensor is stored on CPU.
  void *host_ptr_;
  // File name when the KernelTensor is stored on Disk.
  std::string file_name_;
  // Token for unfinished async io.
  std::optional<size_t> aio_token_;
  // Mark which heterogeneous resource should be allocated.
  NeedAllocateHeteRes need_alloc_hete_res_{NeedAllocateHeteRes::NoNeedHeteRes};
  std::string ToString() {
    std::ostringstream ofs;
    ofs << this << " host ptr:" << host_ptr_ << " file name:" << file_name_
        << " need alloc hete res:" << need_alloc_hete_res_;
    return ofs.str();
  }
};
using HeterogeneousInfoPtr = std::shared_ptr<HeterogeneousInfo>;
namespace device {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
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
  using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
  DeviceAddress();
  explicit DeviceAddress(const AddressCommonPtr &address_common);

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

  virtual std::string ToString() const;

  virtual void CloneDeviceAddress(const DeviceAddressPtr &device_address);

  virtual DeviceAddressPtr CloneDeviceAddress() { MS_LOG(EXCEPTION) << "Not implemented."; }

  virtual bool CopyDeviceToHostWithoutSyncStream(void *dst, size_t dst_size, const void *src, size_t src_size) {
    return true;
  }
  virtual bool AsyncHostToDevice(size_t size, TypeId /* type */, const void *host_ptr,
                                 size_t stream_id = SIZE_MAX) const {
    return true;
  }
  virtual bool AsyncHostToDevice(size_t size, TypeId type, const tensor::TensorDataPtr &tensor_data,
                                 const std::string &format, size_t stream_id = SIZE_MAX) const {
    return true;
  }

  virtual bool AsyncHostToDevice(size_t size, const void *host_ptr, size_t stream_id = SIZE_MAX) const { return true; }
  virtual bool AsyncDeviceToHost(size_t size, void *host_ptr, size_t stream_id = SIZE_MAX) const { return true; }

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
  virtual bool AsyncDeviceToDevice(const DeviceAddress *, size_t stream_id = SIZE_MAX) const { return true; }
  virtual bool CopyDeviceToHost(void *dst, const void *src, const size_t &size) const { return true; }
  virtual bool CopyHostToDevice(void *dst, const void *src, const size_t &size) const { return true; }

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
  void set_device_name(const std::string &device_name);

  uint32_t device_id() const;
  void set_device_id(uint32_t device_id);

  void set_stream_id(uint32_t stream_id);
  const uint32_t stream_id() const;

  bool managed_by_somas() const;
  void set_managed_by_somas(bool managed_by_somas);

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node);
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const;
  void ClearHeldByNodes();

  virtual void SetNodeIndex(const AnfNodePtr &node, size_t out_index);
  KernelWithIndex GetNodeIndex() const;

  void IncreaseNewRefCount(const std::string &op_name, size_t i = 1);
  void IncreaseNewRefCount(size_t i = 1);
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
  bool IsNotNeedAllocWOLock() const;

  using SyncUserDataHandler = void (*)(DeviceAddress *const device_address);
  // Return the valid device ptr.
  virtual void *GetValidPtr(size_t);

  inline void TouchSyncHandler() {
    if (!need_sync_user_data_ || user_data() == nullptr) {
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

  HeterogeneousInfoPtr heterogeneous_info() const;
  void set_heterogeneous_info(HeterogeneousInfoPtr hete_info);

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

  void set_ref_count_without_hold(const PointerRefCountPtr &ptr_ref_cnt);

  void set_is_view(bool is_view);
  bool is_view() const;
  AddressCommonPtr address_common() const;
  void set_address_common(const AddressCommonPtr &address_common);
  ContinuousDeviceAddressesPtr continuous_device_addresses() const;
  void set_continuous_device_addresses(const ContinuousDeviceAddressesPtr &continuous_device_addresses);
  size_t size() const { return address_common_->size_; }

 protected:
  // address basic info
  AddressCommonPtr address_common_{nullptr};

  void *GetDevicePtr() const { return address_common_->pointer_ref_count_->ptr(); }
  void SetDevicePtr(void *ptr) const { address_common_->pointer_ref_count_->set_ptr(ptr); }

  void SetTypeId(TypeId type) const { address_common_->dtype_id_ = type; }
  virtual bool AsyncDeviceToDevice(const ShapeVector &, size_t, TypeId, const void *, const std::string &,
                                   size_t stream_id = SIZE_MAX) const {
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

  // Move to kernel tensor later.
  // host_shape_/hete_info_/user_data_ will be removed from device address later.
  // The flatten shape(maybe after padding) vector.
  // Note: the 'host_shape_' will be repalced by 'shape_vector_' in the future.
  ShapeVector host_shape_{};
  // heterogeneous info
  HeterogeneousInfoPtr hete_info_{nullptr};
  UserDataPtr user_data_{nullptr};

  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUMemoryManager;
  friend class mindspore::device::cpu::CPUDeviceContext;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::gpu::GPUDeviceContext;
  friend class mindspore::device::ascend::AscendRuntimeCore;
  friend class mindspore::device::ascend::AscendMemoryManager;
  friend class mindspore::device::ascend::DataDumper;
  friend class mindspore::SingleOpInferSession;
  friend class mindspore::RuntimeUtils;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
